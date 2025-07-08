from typing import Tuple

import networkx as nx

import egr.glasgow_subgraph_solver as gss
from egr.fsg import gaston
from egr.graph_utils import get_neighborhood_subgraph


class SubgraphIsomorphism:
    def __init__(self, G, H):
        self.G = G
        self.H = H

    def is_isomorphic(self, index: int, n: int) -> Tuple[int, bool]:
        hops: int = nx.eccentricity(self.H, self.H.graph[gaston.rootAttr])
        G = get_neighborhood_subgraph(self.G.copy(), index, hops)
        gaston.makeRootNode(G, index)
        return n, subgraph_is_isomorphic(G, self.H.copy())


def make_feature(G_target: nx.Graph, G_pattern: nx.Graph) -> float:
    return 1.0 if subgraph_is_isomorphic(G_target, G_pattern) else 0.0


def subgraph_is_isomorphic(G_target: nx.Graph, G_pattern: nx.Graph) -> bool:
    return gss.subgraph_is_isomorphic(G_target, G_pattern)
