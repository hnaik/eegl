import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict

from egr.log import init_logging
from egr.util import load_graph, save_json

LOG = logging.getLogger('convert')


def is_m0(node_id: int) -> True:
    return (node_id % 5) == 0


def is_m1(node_id: int) -> True:
    return (node_id % 5) == 1


def is_b0(node_id: int) -> True:
    return (node_id % 5) == 3


def is_b1(node_id: int) -> True:
    return (node_id % 5) == 2


def is_t(node_id: int) -> True:
    return (node_id % 5) == 4


class LabelMaker:
    @staticmethod
    def make_v0(args):
        labels = []
        for i in range(args.nodes):
            if i < args.random_nodes:
                labels.append(0)
            else:
                if is_m0(i) or is_m1(i):
                    labels.append(1)
                elif is_b0(i) or is_b1(i):
                    labels.append(2)
                elif is_t(i):
                    labels.append(3)
                else:
                    raise RuntimeError('Unknown')
        return labels


def main(args):
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    fn = getattr(LabelMaker, f'make_{args.variant}')
    labels = fn(args)
    with args.output_file.open('w') as f:
        LOG.info('Saving to %s', args.output_file)
        f.write(','.join([str(i) for i in labels]))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--nodes', type=int, required=True)
    parser.add_argument('--random-nodes', type=int, required=True)
    parser.add_argument('--output-file', type=Path, required=True)
    parser.add_argument(
        '--variant',
        choices=['v0', 'v1', 'v2', 'v3', 'v4', 'v5'],
        required=True,
    )

    args = parser.parse_args()
    init_logging(level_name='info')
    main(args)
