import logging
import pickle
from argparse import ArgumentParser
from pathlib import Path

import networkx as nx

import egr
from egr.datasets.fullerenes import converter as cvt

LOG = logging.getLogger('process_fullerenes')

# example link to the fullerenes dataset
# https://nanotube.msu.edu/fullerene/C720/C720-0.xyz
# ROOT_URL = 'https://nanotube.msu.edu/fullerene/'
FULLERENES = [
    {'name': 'C180', 'variant': 'C180-0', 'atoms': 180},
    {'name': 'C240', 'variant': 'C240-0', 'atoms': 240},
    {'name': 'C260', 'variant': 'C260-0', 'atoms': 260},
    {'name': 'C320', 'variant': 'C320-0', 'atoms': 320},
    {'name': 'C500', 'variant': 'C500-0', 'atoms': 500},
    {'name': 'C540', 'variant': 'C540-0', 'atoms': 540},
    {'name': 'C720', 'variant': 'C720-0', 'atoms': 720},
]


def main(args):
    process_fullerene(args)


def process_fullerene(args):
    args.output_root.mkdir(parents=True, exist_ok=True)
    for fullerene in FULLERENES:
        name, variant = fullerene['name'], fullerene['variant']
        path = args.output_root / f'{variant}.pkl'

        if path.exists() and not args.overwrite:
            LOG.info('Skipping %s, already exists', path)
            continue
        G = cvt.make_graph_data(args, name, variant)
        LOG.info('Saving graph to %s', path)
        pickle.dump(G, path.open('wb'))


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument(
        '--log-level',
        type=str,
        default='info',
        choices=['debug', 'info', 'warning', 'error', 'critical'],
    )
    parser.add_argument('--temp-dir', type=Path, default='dataset/fullerenes')
    parser.add_argument(
        '--output-root', type=Path, default='dataset/pickled/fullerenes'
    )
    parser.add_argument('--overwrite', action='store_true')
    parsed_args = parser.parse_args()
    egr.init_logging(level_name=parsed_args.log_level)
    main(parsed_args)
