"""
```shell
$ python apps/mk_orbit_labels.py \
    --variant=f13 \
    --output-file=/data/results/input_data/f12/labels.txt
```
"""

import logging
from argparse import ArgumentParser
from typing import Callable
from pathlib import Path

from egr.log import init_logging

LOG = logging.getLogger('template')


def make_m2_labels():
    labels = [0] * 700
    for i in range(300, 700, 5):
        labels[i] = 1
        labels[i + 1] = 2
        labels[i + 2] = 3
        labels[i + 3] = 2
        labels[i + 4] = 4
    return labels


def make_m3_labels():
    return make_m2_labels()


def make_m4_labels():
    return make_m2_labels()


def make_m5_labels():
    labels = [0] * 780
    for i in range(300, 700, 5):
        labels[i] = 1
        labels[i + 1] = 2
        labels[i + 2] = 3
        labels[i + 3] = 4
        labels[i + 4] = 5
    for i in range(700, 780):
        labels[i] = 4
    return labels


def make_m6_labels():
    labels = [0] * 780
    for i in range(300, 700, 5):
        labels[i] = 1
        labels[i + 1] = 2
        labels[i + 2] = 3
        labels[i + 3] = 4
        labels[i + 4] = 5
    for i in range(700, 780):
        labels[i] = 3
    return labels


def main(args):
    func: Callable = globals()[f'make_{args.variant}_labels']
    labels = func()
    LOG.info('Saving labels to %s', args.output_file)
    args.output_file.open('w').write(','.join([str(i) for i in labels]))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output-file', type=Path, required=True)
    parser.add_argument(
        '--variant',
        type=str,
        required=True,
        choices=[f'm{i}' for i in range(1, 7)] + ['m1_bridged'],
    )
    parser.add_argument(
        '-l',
        '--log-level',
        type=str,
        default='info',
        choices=['debug', 'info', 'warning', 'error', 'critical'],
    )
    args = parser.parse_args()
    init_logging(args.log_level)
    main(args)
