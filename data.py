import os
import json
import torch
import multiprocessing
import numpy as np
from pathlib import Path
from collections import Counter
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight

__all__ = ['load_news_category']


class _NewsCategoryDataset(Dataset):
    def __init__(self, fpath: str, tokenizer: BertTokenizer):
        self.fpath = Path(fpath)
        self.tokenizer = tokenizer
        self.max_len = 40
        self.all_token_ids, self.all_segments, self.all_attention_mask, self.labels = self._init_dataset()

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx], self.all_attention_mask[idx]), self.labels[idx]

    def __len__(self):
        return len(self.all_token_ids)

    def _init_dataset(self):
        dataset = _load_news_category_dataset(self.fpath)
        label_set = {}

        # features
        features = dataset[0]
        all_token_ids, all_segments, all_attention_mask = self._preprocess(features)

        # labels
        labels = dataset[-1]
        for idx, label in enumerate(sorted(set(labels))):
            label_set[label] = idx
        self._save_label_set(label_set)
        labels = list(map(lambda key: label_set[key], labels))

        return all_token_ids, all_segments, all_attention_mask, torch.tensor(labels)

    def _save_label_set(self, label_set):
        with open(self.fpath.parent / "label_set.json", "w", encoding="utf-8") as f:
            json.dump(label_set, f, ensure_ascii=False, indent=4)

    def _preprocess(self, features):
        pool = multiprocessing.Pool(4)
        out = pool.map(self._mp_worker, features)

        all_token_ids = []
        all_segments = []
        all_attention_mask = []
        for token_ids, segments, attention_mask in out:
            all_token_ids.append(token_ids)
            all_segments.append(segments)
            all_attention_mask.append(attention_mask)
        return (
            torch.tensor(all_token_ids, dtype=torch.long),
            torch.tensor(all_segments, dtype=torch.long),
            torch.tensor(all_attention_mask, dtype=torch.long),
        )

    def _mp_worker(self, feature):
        out = self.tokenizer(feature, truncation=True, padding='max_length', max_length=self.max_len)
        return out["input_ids"], out["token_type_ids"], out["attention_mask"]


def _split_new_category_dataset(fpath: Path, fname: str):
    # Count the original labels
    labels = []
    with open(fpath, 'r', encoding='utf-8') as file:
        for idx, line in enumerate(file):
            line = json.loads(line)
            if len(line["short_description"]) == 0:
                continue
            labels.append(line["category"])
    labels_count = dict(Counter(labels))

    # split dataset
    train, test = [], []
    vel_count = {key: int(labels_count[key] * 0.1) for key in labels_count}
    _vel_count = {key: 0 for key in vel_count}
    with open(fpath, 'r', encoding='utf-8') as file:
        for line in file:
            label = json.loads(line)["category"]
            if _vel_count[label] <= vel_count[label]:
                test.append(line)
                _vel_count[label] += 1
            else:
                train.append(line)

    # save
    with open(fpath.parent / f"{fname}_train.json", "w", encoding="utf-8") as f:
        f.write("".join(train))
    with open(fpath.parent / f"{fname}_test.json", "w", encoding="utf-8") as f:
        f.write("".join(test))


def _load_news_category_dataset(fpath: Path):
    def _format(line):
        line = line.replace("\'", "'")
        line = line.replace('“', '"').replace('”', '"').strip()
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]
        return line

    # load News_Category_Dataset_v3.json
    features, labels = [], []  # short_description, category
    with open(fpath, 'r', encoding='utf-8') as file:
        for line in file:
            line = json.loads(line)
            short_description = _format(line["short_description"])
            if len(short_description) == 0:
                continue
            features.append(short_description)
            labels.append(line["category"])
    return features, labels


def load_news_category(fpath: str, local_model_path: str, batch_size: int, class_weights=False):
    """
    ARGS:
        - fpath: path to News_Category_Dataset_v3.json
    """
    fpath = Path(fpath)
    fname = fpath.stem
    if fpath.suffix != ".json":
        raise ValueError("We only accept JSON files.")
    split_dataset = True
    num_workers = os.cpu_count() // 2

    for item in fpath.parent.iterdir():
        if item.name == f"{fname}_train.json" or item.name == f"{fname}_test.json":
            split_dataset = False
            break
    if split_dataset:
        _split_new_category_dataset(fpath, fname)
        print(f"Split {fname}_train.json and {fname}_text.json from {fname}")

    tokenizer = BertTokenizer.from_pretrained(local_model_path)

    train_set = _NewsCategoryDataset(fpath.parent / f"{fname}_train.json", tokenizer)
    train_iter = DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers)
    test_set = _NewsCategoryDataset(fpath.parent / f"{fname}_test.json", tokenizer)
    test_iter = DataLoader(test_set, batch_size, shuffle=False, num_workers=num_workers)

    if not class_weights:
        return train_iter, test_iter
    else:
        _, labels = _load_news_category_dataset(fpath.parent / f"{fname}_train.json")

        # read label_set.json
        with open(fpath.parent / "label_set.json", "r", encoding="utf-8") as f:
            label_set = json.load(f)
        labels = list(map(lambda key: label_set[key], labels))

        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float)

        return train_iter, test_iter, class_weights
