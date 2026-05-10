from torch import nn
from transformers import BertModel


__all__ = ["BERTClassifier"]


class BERTClassifier(nn.Module):
    def __init__(self, local_model_path: str, num_hiddens: int = 256, num_classes: int = 42):
        super().__init__()
        self.bert = BertModel.from_pretrained(local_model_path)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Sequential(
            nn.Linear(768, num_hiddens), nn.ReLU(), nn.Dropout(0.2), nn.Linear(num_hiddens, num_classes)
        )

    def forward(self, tokens_X, segments_X, valid_lens_X):
        outputs = self.bert(input_ids=tokens_X, token_type_ids=segments_X, attention_mask=valid_lens_X)
        # resturn self.output(output.pooler) # pooler
        return self.output(self.dropout(outputs.last_hidden_state[:, 0, :]))
