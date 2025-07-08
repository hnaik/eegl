import logging
import typing as ty
from argparse import ArgumentParser
from pathlib import Path

import networkx as nx

import egr.util as eu
from egr.log import init_logging

LOG = logging.getLogger('template')


def make_lad(input_path: Path) -> ty.Dict:
    return {}


def main(args):
    LOG.info('Starting app with log-level %s', args.log_level)
    lad_data: ty.Dict = make_lad()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-l',
        '--log-level',
        type=str,
        default='info',
        choices=['debug', 'info', 'warning', 'error', 'critical'],
    )
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    init_logging(args.log_level)
    main(args)
