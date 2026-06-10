import os
import json
import click
import torch
import random

import numpy as np
import multiprocessing as mp
import nlpaug.augmenter.word as naw

from tqdm import tqdm
from pathlib import Path
from collections import Counter
from transformers import BertTokenizer, pipeline
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight

__all__ = ["LoadDataset"]


def get_lable_map(fpath, labels):
    """记录babel与idx对应关系"""
    label_map = {}

    # save
    for idx, label in enumerate(sorted(set(labels))):
        label_map[label] = idx
    with open(fpath.parent / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=4)

    return label_map


def split_dataset(features, labels):
    """拆分得到训练集和验证集"""
    # Count the original labels
    labels_count = dict(Counter(labels))

    # split dataset
    train_features, train_labels, test_features, test_labels = [], [], [], []
    vel_count = {
        key: int(labels_count[key] * 0.1) for key in labels_count
    }  # 此处有缺陷: 如果数量为个位数会有问题，在 News_Category_Dataset_v3 未体现
    _vel_count = {key: 0 for key in vel_count}

    for idx, label in enumerate(labels):
        if _vel_count[label] <= vel_count[label]:
            test_features.append(features[idx])
            test_labels.append(label)
            _vel_count[label] += 1
        else:
            train_features.append(features[idx])
            train_labels.append(label)

    return train_features, train_labels, test_features, test_labels


def format_short_description(line):
    line = line.replace("\'", "'")
    line = line.replace('“', '"').replace('”', '"').strip()
    if line.startswith('"') and line.endswith('"'):
        line = line[1:-1]
    return line


def parse_dataset(fpath: Path, tokenizer: str):
    """读取 News_Category_Dataset_v3.json 将其拆分为训练集和验证集，并保存为 pt 文件。"""
    fname = fpath.stem
    if fpath.name != "News_Category_Dataset_v3.json":
        raise ValueError("请指定 News_Category_Dataset_v3.json 具体路径。")

    # 读取 features & labels
    all_features, all_labels = [], []
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            short_description = format_short_description(line["short_description"])
            if len(short_description) == 0:
                continue
            all_features.append(short_description)
            all_labels.append(line["category"])

    label_map = get_lable_map(fpath, all_labels)
    train_features, train_labels, test_features, test_labels = split_dataset(all_features, all_labels)

    # train dataset
    train_set = NewsCategoryDataset((train_features, train_labels), tokenizer, label_map)
    torch.save(
        {
            "token_ids": train_set.all_token_ids,
            "segments": train_set.all_segments,
            "attention_mask": train_set.all_attention_mask,
            "labels": train_set.labels,
        },
        fpath.parent / "train.pt",
    )

    # test dataset
    test_set = NewsCategoryDataset((test_features, test_labels), tokenizer, label_map)
    torch.save(
        {
            "token_ids": test_set.all_token_ids,
            "segments": test_set.all_segments,
            "attention_mask": test_set.all_attention_mask,
            "labels": test_set.labels,
        },
        fpath.parent / "test.pt",
    )

    # class weights
    all_labels_idx = list(map(lambda key: label_map[key], all_labels))

    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels_idx), y=all_labels_idx)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    torch.save(class_weights, fpath.parent / "class_weights.pt")


@click.command()
@click.option('-f', '--fpath', help='News_Category_Dataset_v3.json 文件完整路径')
@click.option('-t', '--tokenizer', help='bert-base-uncased 目录')
def main(fpath, tokenizer):
    mp.set_start_method('spawn', force=True)
    parse_dataset(Path(fpath), tokenizer)


class EDA:
    def __init__(self, back_translate_prob=0.2):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.en2zh = pipeline("translation", model="/root/models/opus-mt-en-zh", device=device)
        self.zh2en = pipeline("translation", model="/root/models/opus-mt-zh-en", device=device)
        self.back_translate_prob = back_translate_prob

    def __call__(self, feature):
        if random.random() < self.back_translate_prob:
            zh_text = self.en2zh(feature)[0]['translation_text']
            return self.zh2en(zh_text)[0]['translation_text']

        op = random.choice(["None", "delete", "swap"])
        if op == "None":
            return feature
        else:
            aug = naw.RandomWordAug(action=op, aug_min=2, aug_max=5)
        return aug.augment(feature)[0]


class NewsCategoryDataset(Dataset):
    def __init__(self, dataset: tuple[list, list], tokenizer: str, label_map: dict, is_train: bool = True):
        self.features, self.labels = dataset
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.is_train = is_train
        self.eda = None

        self.all_token_ids, self.all_segments, self.all_attention_mask, self.labels = self.init_dataset()

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx], self.all_attention_mask[idx]), self.labels[idx]

    def __len__(self):
        return len(self.all_token_ids)

    def init_dataset(self):
        # features
        all_token_ids, all_segments, all_attention_mask, all_labels = self._preprocess(self.features, self.labels)

        # labels
        all_labels = list(map(lambda key: self.label_map[key], all_labels))

        return all_token_ids, all_segments, all_attention_mask, torch.tensor(all_labels)

    def _preprocess(self, features, labels):
        all_token_ids = []
        all_segments = []
        all_attention_mask = []
        all_labels = []

        with mp.Pool(processes=4, initializer=self._init_worker, initargs=(self.tokenizer, self.is_train)) as pool:
            results = list(
                tqdm(pool.imap(self._worker_without_eda, features), total=len(features), desc="Preprocessing")
            )

        _all_token_ids, _all_segments, _all_attention_mask = zip(*results)

        for idx, value in enumerate(_all_token_ids):
            all_token_ids.extend(value)
            all_segments.extend(_all_segments[idx])
            all_attention_mask.extend(_all_attention_mask[idx])
            all_labels.extend([labels[idx]] * len(value))

        return (
            torch.tensor(all_token_ids, dtype=torch.long),
            torch.tensor(all_segments, dtype=torch.long),
            torch.tensor(all_attention_mask, dtype=torch.long),
            all_labels,
        )

    @staticmethod
    def _init_worker(tokenizer, is_train):
        global worker_eda, worker_tokenizer, worker_max_len, work_is_train

        worker_eda = EDA(back_translate_prob=0.1)
        worker_tokenizer = BertTokenizer.from_pretrained(tokenizer)
        worker_max_len = 40
        work_is_train = is_train

    @staticmethod
    def _worker_func(feature):
        global worker_eda, worker_tokenizer, worker_max_len, work_is_train

        _feature = [feature]

        if work_is_train:
            new_feature = worker_eda(feature)
            if new_feature != feature:
                _feature.append(new_feature)

        out = worker_tokenizer(_feature, truncation=True, padding='max_length', max_length=worker_max_len)
        return out["input_ids"], out["token_type_ids"], out["attention_mask"]

    @staticmethod
    def _worker_without_eda(feature):
        global worker_eda, worker_tokenizer, worker_max_len

        _feature = [feature]  # 保持与有eda时一致，会影响输出

        out = worker_tokenizer(_feature, truncation=True, padding='max_length', max_length=worker_max_len)
        return out["input_ids"], out["token_type_ids"], out["attention_mask"]


class LoadDataset(Dataset):
    def __init__(self, data):
        self.all_token_ids = data["token_ids"]
        self.all_segments = data["segments"]
        self.all_attention_mask = data["attention_mask"]
        self.labels = data["labels"]

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx], self.all_attention_mask[idx]), self.labels[idx]

    def __len__(self):
        return len(self.all_token_ids)


if __name__ == "__main__":
    # export LD_LIBRARY_PATH=/root/miniconda3/envs/bert-classifier/lib:$LD_LIBRARY_PATH
    # python data.py -f "/root/autodl-tmp/BertClassifier/dataset/News_Category_Dataset_v3.json" -t "/root/models/bert-base-uncased"
    main()
