import logging
import pathlib
import typing as ty
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import egr.util as util
from egr.log import init_logging

LOG = logging.getLogger('create_index')


def write(train: np.array, test: np.array, path: Path):
    val_size: int = test.shape[0] // 2
    data = dict(train=train, val=test[:val_size], test=test[-val_size:])
    LOG.info('Saving to %s', path)
    util.save_json(data, path)


def make_folds(
    labels: ty.List,
    n_splits: int = 10,
    shuffle: bool = True,
    random_state: int = 42,
    val_idx: bool = True,
) -> ty.List:
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )
    indices = list(range(len(labels)))

    def f(train, rest, labels) -> ty.Dict:
        if val_idx:
            return {'train': train.tolist(), **make_val_test(labels, rest)}
        return {'train': train.tolist(), 'test': rest.tolist()}

    return [f(tr, rest, labels) for tr, rest in skf.split(indices, labels)]


def make_val_test(labels, rest_idx):
    data = {label: [] for label in sorted(set(labels))}
    for idx in rest_idx:
        data[labels[idx]].append(idx)

    fold_data = dict(val=[], test=[])
    for indices in data.values():
        for n, idx in enumerate(indices):
            key = 'val' if n % 2 == 1 else 'test'
            fold_data[key].append(idx)

    rng = np.random.default_rng()
    rng.shuffle(fold_data['val'])
    rng.shuffle(fold_data['test'])
    return fold_data


def main(args):
    indices = np.arange(args.count)

    spec = util.load_variant_spec(args.variant.lower())

    y = np.array([int(e) for e in args.labels_file.open().read().split(',')])
    unique_labels = np.unique(y).tolist()
    skf = StratifiedKFold(n_splits=spec.details.folds, shuffle=True)
    index_dir = args.labels_file.parent / 'indices'
    LOG.info('Creating index directory: %s', index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    split_desc = []
    for i, (train_index, test_index) in enumerate(skf.split(indices, y)):
        fold = i + 1
        file_path = index_dir / f'{fold:02d}.json'
        write(train_index, test_index, file_path)

        tr_labels, tr_counts = np.unique(y[train_index], return_counts=True)
        te_labels, te_counts = np.unique(y[test_index], return_counts=True)

        for i, label in enumerate(unique_labels):

            def safe_get(arr, i) -> int:
                if len(arr) <= i:
                    return 0
                return arr[i]

            split_desc.append(
                dict(
                    fold=fold,
                    train_label=safe_get(tr_labels, i),
                    train_count=safe_get(tr_counts, i),
                    test_label=safe_get(te_labels, i),
                    test_count=safe_get(te_counts, i),
                )
            )
    df = pd.DataFrame(split_desc)
    print(df)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-l',
        '--log-level',
        type=str,
        default='info',
        choices=['debug', 'info', 'warning', 'error', 'critical'],
    )
    parser.add_argument('--count', type=int)
    parser.add_argument('--train-fraction', type=float, default=0.8)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--labels-file', type=pathlib.Path)

    args = parser.parse_args()
    init_logging(args.log_level)
    main(args)


# def make_balanced(args):
#     LOG.info('labels file path: %s', args.labels_file)
#     labels: ty.List = [
#         int(e) for e in args.labels_file.open().read().split(',')
#     ]
#     total_count: int = len(labels)
#     LOG.info('labels: %s', labels)
#     LOG.info('type: %s', type(labels))

#     from collections import Counter

#     counter = Counter(labels)
#     LOG.info('counter: %s', counter)

#     indices = {item: [] for item in sorted(set(labels))}
#     for index, label in enumerate(labels):
#         indices[label].append(index)
#     LOG.info('indices: %s', indices)

#     rng = np.random.default_rng()
#     for label_key in indices:
#         rng.shuffle(indices[label_key])

#     rng = np.random.default_rng()

#     index_dir = args.labels_file.parent / 'indices'
#     LOG.info('Creating index directory: %s', index_dir)
#     index_dir.mkdir(parents=True, exist_ok=True)
#     for fold in range(args.folds.begin, args.folds.end + 1):
#         train_indices: ty.List = []
#         val_indices: ty.List = []
#         test_indices: ty.List = []
#         for label, index_list in indices.items():
#             train_size: int = int(len(index_list) * args.train_fraction)
#             test_size: int = (len(index_list) - train_size) // 2
#             indexes = np.roll(index_list, (fold - 1) * test_size)

#             train_indices.extend(indexes[:train_size])
#             val_indices.extend(indexes[train_size:-test_size])
#             test_indices.extend(indexes[-test_size:])

#         data = dict(
#             train=sorted(train_indices),
#             val=sorted(val_indices),
#             test=sorted(test_indices),
#         )

#         file_path: pathlib.Path = index_dir / f'{fold:02d}.json'
#         LOG.info('Saving index file: %s', file_path)
#         util.save_json(data, file_path)


# def main(args):
#     if args.labels_file:
#         make_balanced(args)
#     else:
#         make_unbalanced(args)


# def make_unbalanced(args):
#     rng = np.random.default_rng()
#     indices = np.arange(args.count)
#     train_size: int = int(indices.shape[0] * args.train_fraction)
#     test_size: int = (indices.shape[0] - train_size) // 2

#     rng.shuffle(indices)
#     LOG.info('num indices: %s', indices.shape[0])
#     output_dir = args.output_root / f'{args.count}'
#     output_dir.mkdir(parents=True, exist_ok=True)
#     LOG.info('args=%s', args)
#     # for i in range(args.folds.begin, args.folds.end + 1):
#     for i in range(1, 11):
#         indices = np.roll(indices, test_size)
#         output_path: Path = output_dir / f'{i:02d}.json'
#         write(indices, train_size, test_size, output_path)
