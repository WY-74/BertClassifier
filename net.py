import torch

from torch import nn
from torch.nn import functional as F
from transformers import BertModel


__all__ = ["BERTClassifier"]


class BERTClassifier(nn.Module):
    def __init__(self, local_model_path: str, dropout: float = 0.2):
        super().__init__()
        self.bert = BertModel.from_pretrained(local_model_path)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 42)
        )

    def forward(self, tokens_X, segments_X, valid_lens_X, labels=None, mixup_alpha=0.2):
        outputs = self.bert(input_ids=tokens_X, token_type_ids=segments_X, attention_mask=valid_lens_X)
        
        if self.training and labels is not None:
            mixed_last_hidden_state, mixed_labels = self.text_mixup(outputs.last_hidden_state[:, 0, :], labels, mixup_alpha)
            return self.output(self.dropout(mixed_last_hidden_state)), mixed_labels
            
        return self.output(self.dropout(outputs.last_hidden_state[:, 0, :]))

    def text_mixup(self, embeddings, labels, alpha=0.2):
        batch_size = embeddings.size(0)
        lam = torch.distributions.Beta(alpha, alpha).sample()
        
        index = torch.randperm(batch_size)  # 随机打乱
        labels = F.one_hot(labels, num_classes=42).float()
        
        mixed_embeddings = lam * embeddings + (1 - lam) * embeddings[index]
        mixed_labels = lam * labels + (1 - lam) * labels[index]
        
        return mixed_embeddings, mixed_labels
