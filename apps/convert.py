import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict

import networkx as nx

from egr.log import init_logging
from egr.util import load_graph, save_json

LOG = logging.getLogger('convert')


def make_compact_data(input_path: Path) -> Dict:
    G = load_graph(input_path)
    u, v = list(zip(*[e for e in G.edges()]))
    return dict(n=G.number_of_nodes(), u=u, v=v)


def main(args):
    for idx, input_file in enumerate(
        sorted(args.input_dir.rglob('**/*.json'))
    ):
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = args.output_dir / f'{idx + 1:04d}.json'
        data = make_compact_data(input_file)
        LOG.info('input_file=%s, output_file=%s', input_file, output_file)
        save_json(data, output_file, separators=(',', ':'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)

    args = parser.parse_args()
    init_logging(level_name='info')
    main(args)
