import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import numpy as np

from egr.log import init_logging
from egr.util import save_features

LOG = logging.getLogger('label_features')


def make_features(path: Path, num_dim: int) -> np.array:
    LOG.info('Creating features from %s', path)
    labels = [int(l) for l in path.read_text().split(',')]
    rows = len(labels)
    X = np.zeros([rows, num_dim])
    for i, label in enumerate(labels):
        X[i, label] = 1
    return X


def main(args):
    output_dir: Path = Path(f'/data/results/input_data/features')
    size = (args.num_nodes, args.num_dim)
    for sample_id in range(args.num_samples):
        path = (
            output_dir
            / f'random_features-{args.num_nodes}-{sample_id + 1:04d}.npy'
        )
        r = np.random.normal(loc=0, scale=1, size=size)
        X = np.clip(r, a_min=0.0, a_max=1.0)
        LOG.info('Saving features to: %s', path)
        save_features(path, X)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-l',
        '--log-level',
        choices=['debug', 'info', 'warning', 'error', 'fatal'],
        default='info',
    )
    parser.add_argument('--num-dim', type=int, default=10)
    parser.add_argument('--num-nodes', type=int, required=True)
    parser.add_argument('--num-samples', type=int, default=1)
    args = parser.parse_args()
    init_logging(level_name=args.log_level)
    main(args)
