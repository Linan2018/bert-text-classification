import torch
from torch.utils.data import Dataset


class BertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512, predict=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.predict = predict

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        texts = self.texts[index]

        inputs = self.tokenizer.encode_plus(
            texts,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )

        ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
        mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
        token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.long)
        targets = torch.tensor(self.labels[index], dtype=torch.long)

        if not self.predict:
            return {
                'ids': ids,
                'mask': mask,
                'token_type_ids': token_type_ids,
                'targets': targets
            }
        else:
            return {
                'ids': ids,
                'mask': mask,
                'token_type_ids': token_type_ids
            }
