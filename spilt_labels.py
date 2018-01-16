import numpy as np
import pandas as pd

np.random.seed(1)

full_labels = pd.read_csv('data/total_labels.csv')
print(full_labels.head())

gb = full_labels.groupby('filename')
grouped_list = [gb.get_group(x) for x in gb.groups]
num = len(grouped_list)
print(num)

train_index = np.random.choice(len(grouped_list), size=int(num * 4 / 5), replace=False)
test_index = np.setdiff1d(list(range(num)), train_index)
train = pd.concat([grouped_list[i] for i in train_index])
test = pd.concat([grouped_list[i] for i in test_index])

print(len(train), len(test))
train.to_csv('data/train_labels.csv', index=None)
test.to_csv('data/test_labels.csv', index=None)