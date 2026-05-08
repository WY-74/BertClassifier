import json
import multiprocessing
from pathlib import Path
from collections import Counter
from torch.utils.data import Dataset


class NewsCategoryDataset(Dataset):
    def __init__(self, fpath, tokenizer):
        self.fpath = Path(fpath)
        self.tokenizer = tokenizer
        self.max_len = 160
        self.all_token_ids, self.all_segments, self.all_attention_mask, self.labels = self._init_dataset()

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx], self.all_attention_mask[idx]), self.labels[idx]

    def __len__(self):
        return len(self.all_token_ids)

    def _init_dataset(self):
        dataset = self._load_news_category_dataset()
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

    def _load_news_category_dataset(self):
        # load News_Category_Dataset_v3.json
        features, labels = [], []  # short_description, category
        with open(self.fpath, 'r', encoding='utf-8') as file:
            for idx, line in enumerate(file):
                line = json.loads(line)
                short_description = self._format(line["short_description"])
                if len(short_description) == 0:
                    continue
                features.append(short_description)
                labels.append(line["category"])
        return features, labels

    def _format(self, line):
        line = line.replace("\'", "'")
        line = line.replace('“', '"').replace('”', '"').strip()
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]
        return line

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
