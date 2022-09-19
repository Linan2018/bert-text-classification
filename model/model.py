import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, AdamW

from dataset.bert_dataset import BertDataset
from model.bert import Bert


class TextClassifier:
    def __init__(self, num_classes, device="cpu",
                 unfreeze_layers=['layer.10', 'layer.11', 'bert.pooler', 'out.', 'fc.'], **kwargs):
        self.device = device
        self.unfreeze_layers = unfreeze_layers
        self.max_len = kwargs.get('max_len', 512)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.bert = Bert(num_classes=num_classes)

        for name, param in self.bert.named_parameters():
            param.requires_grad = False
            for ele in self.unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break

        self.bert.to(self.device)

    def loss_fn(self, outputs, targets):
        return nn.CrossEntropyLoss()(outputs, targets)

    def accuracy_fn(self, outputs, targets):
        return (outputs.argmax(1) == targets).sum().item() / targets.size(0)

    def predict(self, texts, **kwargs):
        print(f"Predicting total {len(texts)} texts")
        batch_size = kwargs.get('batch_size', 32)
        num_workers = kwargs.get('num_workers', 4)

        dataset = BertDataset(texts, None, self.tokenizer, max_len=self.max_len, predict=True)
        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        self.bert.eval()
        with torch.no_grad():
            predictions = []
            for data in tqdm(data_loader, total=len(data_loader)):
                ids = data['ids'].to(self.device, dtype=torch.long)
                mask = data['mask'].to(self.device, dtype=torch.long)
                outputs = self.bert(ids, mask)
                predictions.extend(outputs.argmax(1).tolist())
            return predictions

    def eval(self, texts, labels, **kwargs):
        print(f"Evaluating total {len(texts)} texts")
        batch_size = kwargs.get('batch_size', 64)
        num_workers = kwargs.get('num_workers', 4)

        dataset = BertDataset(texts, labels, self.tokenizer, max_len=self.max_len)
        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        self.bert.eval()
        with torch.no_grad():
            total_loss = 0
            total_accuracy = 0
            for data in tqdm(data_loader, total=len(data_loader)):
                ids = data['ids'].to(self.device, dtype=torch.long)
                mask = data['mask'].to(self.device, dtype=torch.long)
                targets = data['targets'].to(self.device, dtype=torch.long)
                outputs = self.bert(ids, mask)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
                total_accuracy += self.accuracy_fn(outputs, targets)
            print('eval loss:', total_loss / len(data_loader))
            print('eval accuracy:', total_accuracy / len(data_loader))

    def fit(self, texts, labels, **kwargs):

        max_epoch = kwargs.get('max_epoch', 100)
        batch_size = kwargs.get('batch_size', 32)
        shuffle = kwargs.get('shuffle', True)
        num_workers = kwargs.get('num_workers', 4)
        val_rate = kwargs.get('val_rate', 0.2)
        lr = kwargs.get('lr', 1e-5)

        # split train and val
        train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=val_rate,
                                                                            random_state=1)
        print(f"Train {len(train_texts)} texts, Val {len(val_texts)} texts")

        train_dataset = BertDataset(train_texts, train_labels, self.tokenizer, max_len=self.max_len)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        self.optimizer = AdamW(filter(lambda p: p.requires_grad, self.bert.parameters()), lr=lr, correct_bias=False)

        for epoch in range(max_epoch):
            train_loss = 0
            train_acc = 0
            self.bert.train()
            print('############# Epoch {}: Training Start   #############'.format(epoch))
            for batch_idx, data in tqdm(enumerate(train_loader), total=len(train_loader), desc='Training'):
                ids = data['ids'].to(self.device, dtype=torch.long)
                mask = data['mask'].to(self.device, dtype=torch.long)
                # token_type_ids = data['token_type_ids'].to(self.device, dtype = torch.long)
                targets = data['targets'].to(self.device, dtype=torch.long)

                outputs = self.bert(ids, mask)
                self.optimizer.zero_grad()
                loss = self.loss_fn(outputs, targets)
                # if batch_idx%5000==0:
                #   print(f'Epoch: {epoch}, Training Loss:  {loss.item()}')
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                train_acc += self.accuracy_fn(outputs, targets)
                # print('before loss data in training', loss.item(), train_loss)
                # train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))
            print('train loss:', train_loss / len(train_loader))
            print('train accuracy:', train_acc / len(train_loader))
            self.eval(val_texts, val_labels)
            print('############# Epoch {}: Training End     #############'.format(epoch))


if __name__ == '__main__':
    pass
