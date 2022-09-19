import torch.nn as nn
from transformers import BertModel


class Bert(nn.Module):
    def __init__(self, hidden_size=768, num_classes=2):
        super(Bert, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, tokens, masks):
        _, pooled_output = self.bert(tokens, attention_mask=masks).to_tuple()
        dropout_out = self.dropout(pooled_output)
        fc_out = self.fc(dropout_out)
        return self.softmax(fc_out)
