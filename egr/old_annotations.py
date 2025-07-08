import logging
import networkx as nx
from enum import IntEnum
from typing import List

import numpy as np

LOG = logging.getLogger(__name__)


class Feature(IntEnum):
    TRIANGLE = 0
    CLIQUE_4 = 1
    CLIQUE_5 = 2
    DEGREE = 3
    DEGREE_CENTRALITY = 4
    EIGENVECTOR_CENTRALITY = 5
    CLOSENESS_CENTRALITY = 6
    INFORMATION_CENTRALITY = 7
    SUBGRAPH_CENTRALITY = 8
    PAGERANK = 9


def find_cliques(G: nx.Graph, size: int) -> List[int]:
    cliques = [c for c in nx.enumerate_all_cliques(G) if len(c) == size]
    counts = {n: 0 for n in range(G.number_of_nodes())}
    for n in G.nodes():
        for c in cliques:
            if n in c:
                counts[n] += 1
    return cliques, counts


def make_annotations(G: nx.Graph):
    LOG.debug('Making annotation for graph %s', G)
    triangles = nx.triangles(G)
    _, c4_counts = find_cliques(G, 4)
    _, c5_counts = find_cliques(G, 5)
    degree_centrality = nx.degree_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    information_centrality = nx.information_centrality(G)
    subgraph_centrality = nx.subgraph_centrality(G)
    pagerank = nx.pagerank(G)

    for n in G.nodes():
        f: List = G.nodes[n]['feat']
        f[Feature.TRIANGLE] = triangles[n]
        f[Feature.CLIQUE_4] = c4_counts[n]
        f[Feature.CLIQUE_5] = c5_counts[n]
        f[Feature.DEGREE] = G.degree(n)
        f[Feature.DEGREE_CENTRALITY] = degree_centrality[n]
        f[Feature.EIGENVECTOR_CENTRALITY] = eigenvector_centrality[n]
        f[Feature.CLOSENESS_CENTRALITY] = closeness_centrality[n]
        f[Feature.INFORMATION_CENTRALITY] = information_centrality[n]
        f[Feature.SUBGRAPH_CENTRALITY] = subgraph_centrality[n]
        f[Feature.PAGERANK] = pagerank[n]

    return G


def annotate_level_0(G: nx.Graph) -> nx.Graph:
    labels: List = G.graph['labels']
    feature_dim: int = len(G.nodes[0]['feat'])
    default_node_features: np.ndarray = np.array([0.5] * feature_dim)
    for n in G.nodes():
        x = default_node_features.copy()
        x[labels[n]] = 1
        G.nodes[n]['feat'] = x
    return G
