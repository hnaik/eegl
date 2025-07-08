import logging
from collections import Counter

import torch
from sklearn.model_selection import StratifiedShuffleSplit

LOG = logging.getLogger(__name__)


def split_train_test(data, test_frac: float = 0.2):
    train_index, test_index = split_indices(data.x, data.y, frac=test_frac)

    return {
        'train_index': torch.from_numpy(train_index),
        'test_index': torch.from_numpy(test_index),
        'train_mask': make_mask(train_index, data.y.shape[0]),
        'test_mask': make_mask(test_index, data.y.shape[0]),
    }


def split_train_val_test(data, test_frac: float = 0.2):
    train_index, rest_index = split_indices(data.x, data.y, test_frac)
    x, y = data.x[rest_index], data.y[rest_index]

    val_index, test_index = split_indices(x, y, frac=0.5)

    return {
        'train_index': torch.from_numpy(train_index),
        'val_index': torch.from_numpy(val_index),
        'test_index': torch.from_numpy(test_index),
        'train_mask': make_mask(train_index, data.y.shape[0]),
        'val_mask': make_mask(val_index, data.y.shape[0]),
        'test_mask': make_mask(test_index, data.y.shape[0]),
    }


def split_indices(x, y, frac: float = 0.2):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=frac)
    splitter.get_n_splits(x, y)

    train_index, test_index = next(splitter.split(x, y))
    return train_index, test_index


def make_mask(indices, n):
    mask = torch.BoolTensor([False] * n)
    mask[indices] = True
    return mask
