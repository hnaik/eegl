import logging
import os
import time
import yaml
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp
from typing import List

import networkx as nx
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

from egr.data.io import EgrDenseData
from egr.graph_utils import get_neighborhood_subgraph
from egr.log import init_logging
from egr.subgraph_isomorphism import make_feature
from egr.util import load_graph, save_features, normalize_path

LOG = logging.getLogger('pattern_features')

ATTR = '__root__'


class Config:
    def __init__(self, path):
        self.__dict__.update(yaml.safe_load(path.open()))
        self.root_path = normalize_path(self.root_path)
        self.patterns_dir = self.root_path / 'pattern_graphs'
        if not hasattr(self, 'nproc'):
            self.__dict__.update({'nproc': os.cpu_count()})


class FeatureMaker:
    def __init__(self, G: nx.Graph, subgraphs: List[nx.Graph], dim: int):
        self.G = G
        self.subgraphs = subgraphs
        self.dim = dim

    def make_annotations(self, node_id: int):
        begin = time.time()
        x = self._make_annotations(node_id)
        end = time.time()
        pid = os.getpid()
        LOG.debug('[PID:%5d] n=%3d in %04.2fs', pid, node_id, end - begin)
        return x

    def _make_annotations(self, node_id: int):
        x: np.ndarray = np.zeros(self.dim)
        for i, G_p in enumerate(self.subgraphs):
            radius: int = min(nx.radius(G_p), 3)
            G_s = get_neighborhood_subgraph(
                self.G, root_id=node_id, hops=radius
            )
            G_s.graph[ATTR] = node_id
            G_s.nodes[node_id][ATTR] = True
            x[i] = make_feature(G_s, G_p)
        return x


def make_features(path: Path, subgraphs: List, cfg: Config):
    G: nx.Graph = EgrDenseData.load_graph(path)
    annotator = FeatureMaker(G, subgraphs, cfg.data_dim)
    X = np.zeros([G.number_of_nodes(), cfg.data_dim])
    nodes: List = G.nodes()
    nproc: int = os.cpu_count() if not hasattr(cfg, 'nproc') else cfg.nproc
    with Pool(processes=nproc) as p:
        X_fut = p.map(annotator.make_annotations, nodes)
        pbar = tqdm(nodes)
        for n in pbar:
            X[n, :] = X_fut[n]
            pbar.set_description(f'finished {n:04d}')
    return X


def annotate(cfg: Config):
    for pattern in cfg.pattern_names:
        patterns_dir = cfg.patterns_dir / pattern
        subgraphs = [load_graph(path) for path in patterns_dir.rglob('*.json')]
        for var in cfg.variants:
            LOG.debug('sg=%s, %s', subgraphs, cfg.sample_ids)
            dir_path = cfg.root_path / var
            for sid in cfg.sample_ids:
                input_path: Path = dir_path / f'{sid}.json'
                LOG.info('Annotating %s', input_path)
                X = make_features(input_path, subgraphs, cfg)
                output_dir: Path = dir_path / pattern
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f'{sid}.npy'
                LOG.info('Saving features to %s', output_path)
                save_features(output_path, X)


def main(args):
    cfg = Config(args.config)
    LOG.debug('config=%s', cfg.__dict__)
    annotate(cfg)
    LOG.info('Elapsed time %s', datetime.now() - start)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=Path, required=True)
    parser.add_argument(
        '-l',
        '--log-level',
        choices=['debug', 'info', 'warning', 'error', 'fatal'],
        default='info',
    )
    args = parser.parse_args()
    init_logging(args.log_level)
    start = datetime.now()
    main(args)
