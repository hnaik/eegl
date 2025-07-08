import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

import egr
from egr.util import save_features

LOG = logging.getLogger('template')


def main(args):
    X = np.ones([args.rows, args.cols])
    save_features(args.output_path, X)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-l',
        '--log-level',
        type=str,
        default='info',
        choices=['debug', 'info', 'warning', 'error', 'critical'],
    )
    parser.add_argument('--rows', type=int, required=True)
    parser.add_argument('--cols', type=int, default=10)
    parser.add_argument('--output-path', type=Path, required=True)

    args = parser.parse_args()
    egr.init_logging(args.log_level)
    main(args)
