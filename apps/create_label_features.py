import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import numpy as np

from egr.log import init_logging
from egr.util import save_features

LOG = logging.getLogger('create_label_features')


def make_features(path: Path, num_dim: int) -> np.array:
    labels = [int(l) for l in path.read_text().split(',')]
    rows = len(labels)
    cols = max(num_dim, len(set(labels)))
    X = np.zeros([rows, cols]) + 0.1
    LOG.info('Creating features from %s, dim:%s', path, X.shape)
    for i, label in enumerate(labels):
        X[i, label] = 1
    return X


def main(args):
    X = make_features(args.labels_file, args.num_dim)
    LOG.info('Saving to %s', args.features_file)
    save_features(args.features_file, X)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-l',
        '--log-level',
        choices=['debug', 'info', 'warning', 'error', 'fatal'],
        default='info',
    )
    parser.add_argument('--labels-file', type=Path, required=True)
    parser.add_argument('--features-file', type=Path, required=True)
    parser.add_argument('--num-dim', type=int, default=10)
    args = parser.parse_args()
    init_logging(level_name=args.log_level)
    main(args)
