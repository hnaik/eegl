import networkx as nx

import pytest


@pytest.fixture(scope='function')
def create_motif_test_graph():
    return _create_motif_test_graph


def _create_motif_test_graph():
    G = nx.Graph()
    G.add_edges_from(
        [
            (0, 4),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 9),
            (2, 7),
            (2, 8),
            (2, 11),
            (3, 5),
            (3, 6),
            (3, 7),
            (3, 11),
            (4, 7),
            (5, 6),
            (6, 8),
            (7, 9),
            (7, 10),
            (8, 9),
            (8, 10),
            (8, 11),
        ]
    )
    return G
