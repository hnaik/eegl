import logging
import time
import typing as ty
from dataclasses import dataclass

import networkx as nx
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

import egr.graph_utils as gu
import egr.subgraph_isomorphism as si
import egr.util as util
from egr.fsg import gaston

LOG = logging.getLogger(__name__)


def perform_annotations(G, subgraphs, args, explain_ids, dims):
    annotator = AsyncAnnotator(G, subgraphs, dims)

    n_jobs = -2
    runner = Parallel(n_jobs=n_jobs)

    params = [(node_id, dim) for node_id in explain_ids for dim in range(dims)]

    pbar = tqdm(params)
    result = runner(delayed(annotator)(*args) for args in pbar)

    X = torch.ones(G.number_of_nodes(), dims, dtype=torch.long) * -1
    begin = time.time()
    for annot in result:
        X[annot.node_id, annot.dim] = annot.iso
    end = time.time()

    LOG.info('Finished annotations in %ss', end - begin)
    LOG.info('Saving features %s to %s', X.shape, args.output_feature_file)
    args.output_feature_file.parent.mkdir(parents=True, exist_ok=True)
    util.save_features(args.output_feature_file, X)


@dataclass
class Annotation:
    node_id: int
    dim: int
    iso: float


class AsyncAnnotator:
    def __init__(self, G: nx.Graph, subgraphs: ty.List[nx.Graph], dim: int):
        self.G = G
        self.subgraphs = subgraphs[:dim]
        self.dim = dim

    def __call__(self, node_id: int, idx: int):
        return self.annotate(node_id, idx)

    def annotate(self, node_id: int, idx: int):
        if idx >= len(self.subgraphs):
            return Annotation(node_id, idx, 0.25)
        H: nx.Graph = self.subgraphs[idx]
        pattern_root = H.graph[gaston.rootAttr]
        gaston.makeRootNode(H, pattern_root)

        hops = nx.eccentricity(H, pattern_root)
        G: nx.Graph = gu.get_neighborhood_subgraph(self.G, node_id, hops)
        gaston.makeRootNode(G, node_id)

        return Annotation(node_id, idx, si.make_feature(G, H))
