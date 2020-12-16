import torch
import numpy as np
from dataset.dataset import *

dataset = KrakowDataset()
raw_df = dataset.data.set_index(['sensor_index', 'UTC time'])
long_seq = raw_df.loc[2, '2017-01']['temperature'].dropna()

print(long_seq)


window_size = 12
short_seqs = []
for i in range(long_seq.shape[0] - window_size + 1):
    short_seqs.append(long_seq.iloc[i:i+window_size].tolist())
short_seqs = np.array(short_seqs)
print(short_seqs.shape)

train_set_proportion, val_set_proportion = 0.6, 0.2
total_len = long_seq.shape[0]
train_val_split = int(total_len * train_set_proportion)
val_test_split = int(total_len * (train_set_proportion + val_set_proportion))
train_seq, val_seq, test_seq = long_seq[:train_val_split],\
                               long_seq[train_val_split:val_test_split],\
                               long_seq[val_test_split:]

train_set = []
for i in range(train_seq.shape[0] - window_size):
    train_set.append(train_seq.iloc[i:i+window_size].tolist())
train_set = np.array(train_set)
print(train_set.shape)

