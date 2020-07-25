import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd


def accuracy(labels_test, percept_preds):
    cm = confusion_matrix(labels_test, percept_preds)
    recall = np.diag(cm) / cm.sum(1)
    precision = np.diag(cm) / cm.sum(0)
    F1_percept = 2 * recall.mean() * precision.mean() / (recall.mean() + precision.mean())
    return cm, F1_percept


df = pd.DataFrame(np.random.randn(2,2), columns=['ali', 'moj'])
df = df.append(['sss', 'dsds'])
df.to_csv('ali.csv', index=False)
df2 = pd.read_csv('ali.csv')
print(df2)
print(df)
