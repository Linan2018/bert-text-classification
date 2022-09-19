import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from model.model import TextClassifier

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Load data
    df = pd.read_csv('input/SPAM_text_message.csv')
    texts = df['Message'].tolist()
    le = LabelEncoder()
    labels = le.fit_transform(df['Category'])

    # Train model
    tc = TextClassifier(num_classes=2, device='cuda')
    tc.fit(texts, labels)
