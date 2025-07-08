import networkx as nx
import numpy as np
from torch import LongTensor

from apps.gnn_explainer.gaston_sgm.gaston import makeRootNode
from egr.subgraph_mining import compute_scores, pick_candidates_round_robin
from egr.log import init_logging

import logging

LOG = logging.getLogger('test_subgraph_mining')

init_logging(level_name='debug')


def test_compute_f1_score():
    G = nx.from_edgelist(
        [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (0, 5), (1, 6)]
    )
    indices = [0, 1, 2, 3, 4, 5]
    labels = [1, 1, 2, 2, 3, 0]

    G_triangle = nx.from_edgelist([(0, 1), (1, 2), (2, 0)])
    makeRootNode(G_triangle, 0)

    expected = 0.8
    actual = compute_scores(
        G.copy(), G_triangle, indices, label=1, labels=labels
    )
    assert actual == expected


def test_pick_candidates_round_robin():
    def make_g(name: str):
        G = nx.Graph()
        G.name = name
        return G

    data = {
        0: [
            {'score': 0.99, 'graph': make_g('0_1')},
            {'score': 0.98, 'graph': make_g('0_2')},
            {'score': 0.97, 'graph': make_g('0_3')},
        ],
        1: [
            {'score': 0.99, 'graph': make_g('1_1')},
            {'score': 0.98, 'graph': make_g('1_2')},
            {'score': 0.97, 'graph': make_g('1_3')},
        ],
        2: [
            {'score': 0.99, 'graph': make_g('2_1')},
            {'score': 0.98, 'graph': make_g('2_2')},
            {'score': 0.97, 'graph': make_g('2_3')},
        ],
    }
    expected = [
        make_g('0_1'),
        make_g('1_1'),
        make_g('2_1'),
        make_g('0_2'),
        make_g('1_2'),
    ]
    actual = pick_candidates_round_robin(data, data_dim=5)
    assert [a.name for a in actual] == [e.name for e in expected]


def test_pick_candidates_round_robin_short_label_candidates():
    def make_g(name: str):
        G = nx.Graph()
        G.name = name
        return G

    data = {
        0: [
            {'score': 0.99, 'graph': make_g('0_1')},
            {'score': 0.98, 'graph': make_g('0_2')},
        ],
        1: [
            {'score': 0.99, 'graph': make_g('1_1')},
        ],
        2: [
            {'score': 0.99, 'graph': make_g('2_1')},
            {'score': 0.98, 'graph': make_g('2_2')},
        ],
    }
    expected = [
        make_g('0_1'),
        make_g('1_1'),
        make_g('2_1'),
        make_g('0_2'),
        make_g('2_2'),
    ]
    actual = pick_candidates_round_robin(data, data_dim=5)
    assert [a.name for a in actual] == [e.name for e in expected]


def test_pick_candidates_round_robin_not_enough_candidates():
    def make_g(name: str):
        G = nx.Graph()
        G.name = name
        return G

    data = {
        0: [
            {'score': 0.99, 'graph': make_g('0_1')},
        ],
        1: [
            {'score': 0.99, 'graph': make_g('1_1')},
        ],
        2: [
            {'score': 0.99, 'graph': make_g('2_1')},
        ],
    }
    expected = [make_g('0_1'), make_g('1_1'), make_g('2_1')]
    actual = pick_candidates_round_robin(data, data_dim=5)
    assert len(actual) == 3
    assert [a.name for a in actual] == [e.name for e in expected]
