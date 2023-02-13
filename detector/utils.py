import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

class SequenceDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seqs = seqs
        self.labels = labels
        self.feature_len = seqs.shape[-1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.seqs[i], self.labels[i]


def apply_sliding_window(data, seq_len=10, flatten=False):
    """
    Parameters
    ----------
    data: sequence data
    seq_len: the length of sliding window

    Returns
    -------
    the first: data after being applied sliding window to
    the second: the ground truth; for example the values from t-w to t are the input so the value at t+1 is the ground
    truth.
    """
    seq_ls = []
    label_ls = []

    for i in range(seq_len, len(data)):
        if not flatten:
            seq_ls.append(data[i - seq_len: i])
        else:
            seq_ls.append(data[i - seq_len: i].flatten())
        label_ls.append(data[i])

    return np.array(seq_ls, dtype=np.float32), np.array(label_ls, dtype=np.float32)


def use_mini_batch(data, labels, batch_size, device='cpu'):
    """
    Returns
    -------
    datalodaer is an iterable dataset. In each iteration, it will return a tuple, the first item is the data and the
    second item is the label. So this object is usually used in training a model.
    You can use len() to know the batch count of the dataset
    """
    data = torch.tensor(data)
    labels = torch.tensor(labels)
    seq_dataset = SequenceDataset(data, labels)
    
    if device == 'cpu':
        dataloader = DataLoader(seq_dataset, batch_size=batch_size, drop_last=True)
    else:
        dataloader = DataLoader(seq_dataset, batch_size=batch_size, drop_last=True,
                               num_workers=8, pin_memory=True)

    return dataloader


def train_test_split(data, train_ratio=0.8):
    train_cnt = round(train_ratio * len(data))
    train_data = data[:train_cnt]
    test_data = data[train_cnt:]
    return train_data, test_data
