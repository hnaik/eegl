import typing as ty
from collections import Counter

import numpy as np

import apps.create_index as ci


def test_make_folds():
    n = 57
    k = 5

    labels = [i % 2 for i in range(n)]
    np.random.shuffle(labels)

    folds = ci.make_folds(labels, k)
    assert len(folds) == k

    split_size = n // k
    rem = n % split_size
    train_size = split_size * (k - 1)
    val_size = split_size // 2
    test_size = split_size // 2

    for fold in ci.make_folds(labels, k):
        assert len(fold['train']) in range(train_size, train_size + rem + 1)
        assert len(fold['val']) in range(val_size, val_size + 2)
        assert len(fold['test']) in range(test_size, test_size + 2)

        train_label_counts = Counter([labels[idx] for idx in fold['train']])
        assert train_label_counts[0] in label_range(train_size, rem)
        assert train_label_counts[1] in label_range(train_size, rem)

        val_label_counts = Counter([labels[idx] for idx in fold['val']])
        assert val_label_counts[0] in label_range(val_size)
        assert val_label_counts[1] in label_range(val_size)

        test_label_counts = Counter([labels[idx] for idx in fold['test']])
        assert test_label_counts[0] in label_range(test_size)
        assert test_label_counts[1] in label_range(test_size)


def label_range(size: int, rem: int = 1, n_labels: int = 2) -> ty.Iterable:
    n = size // n_labels
    return range(n, n + rem + 1)
