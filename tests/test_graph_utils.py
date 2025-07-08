import typing as ty
from pathlib import Path


import networkx as nx

from egr import graph_utils as gu


def test_has_element__has():
    graph_list: ty.List[nx.Graph] = [
        nx.cycle_graph(3),
        nx.cycle_graph(4),
        nx.complete_graph(3),
        nx.complete_graph(4),
        nx.complete_graph(5),
    ]

    assert gu.has_element(graph_list, nx.cycle_graph(3))
    assert gu.has_element(graph_list, nx.cycle_graph(4))
    assert gu.has_element(graph_list, nx.complete_graph(3))
    assert gu.has_element(graph_list, nx.complete_graph(4))
    assert gu.has_element(graph_list, nx.complete_graph(5))


def test_has_element__has_not():
    graph_list: ty.List[nx.Graph] = [
        nx.cycle_graph(3),
        nx.cycle_graph(4),
        nx.complete_graph(3),
        nx.complete_graph(4),
        nx.complete_graph(5),
    ]

    assert not gu.has_element(graph_list, nx.cycle_graph(5))
    assert not gu.has_element(graph_list, nx.cycle_graph(6))
    assert not gu.has_element(graph_list, nx.complete_graph(6))
    assert not gu.has_element(graph_list, nx.complete_graph(7))
    assert not gu.has_element(graph_list, nx.complete_graph(8))


def test_subgraph_is_isomorphic():
    G1 = nx.json_graph.node_link_graph(
        {
            'directed': False,
            'multigraph': False,
            'graph': {
                '__root__': 0,
                'label': 0,
                'accuracy': 0.9984,
                'precision': 0.996,
                'recall': 1.0,
                'f1_score': 0.9981515711645101,
                'support': None,
                'f1_score-binary': 0.9981515711645101,
                'num_indices': 630,
                'feature_index': 0,
            },
            'nodes': [
                {'label': 1, '__root__': True, 'id': 0},
                {'label': 0, 'id': 1},
                {'label': 0, 'id': 2},
                {'label': 0, 'id': 3},
                {'label': 0, 'id': 4},
                {'label': 0, 'id': 5},
                {'label': 0, 'id': 6},
                {'label': 0, 'id': 7},
            ],
            'links': [
                {'label': 0, 'source': 0, 'target': 1},
                {'label': 0, 'source': 0, 'target': 3},
                {'label': 0, 'source': 0, 'target': 7},
                {'label': 0, 'source': 1, 'target': 2},
                {'label': 0, 'source': 1, 'target': 4},
                {'label': 0, 'source': 3, 'target': 4},
                {'label': 0, 'source': 3, 'target': 5},
                {'label': 0, 'source': 3, 'target': 6},
            ],
        }
    )

    G2 = nx.json_graph.node_link_graph(
        {
            'directed': False,
            'multigraph': False,
            'graph': {
                '__root__': 0,
                'label': 0,
                'accuracy': 0.9984,
                'precision': 0.996,
                'recall': 1.0,
                'f1_score': 0.9981515711645101,
                'support': None,
                'f1_score-binary': 0.9981515711645101,
                'num_indices': 630,
                'feature_index': 0,
            },
            'nodes': [
                {'label': 1, '__root__': True, 'id': 0},
                {'label': 0, 'id': 1},
                {'label': 0, 'id': 2},
                {'label': 0, 'id': 3},
                {'label': 0, 'id': 4},
                {'label': 0, 'id': 5},
                {'label': 0, 'id': 6},
                {'label': 0, 'id': 7},
            ],
            'links': [
                {'label': 0, 'source': 0, 'target': 1},
                {'label': 0, 'source': 0, 'target': 3},
                {'label': 0, 'source': 0, 'target': 7},
                {'label': 0, 'source': 1, 'target': 2},
                {'label': 0, 'source': 1, 'target': 4},
                {'label': 0, 'source': 3, 'target': 4},
                {'label': 0, 'source': 3, 'target': 5},
                {'label': 0, 'source': 3, 'target': 6},
            ],
        }
    )

    G3 = nx.json_graph.node_link_graph(
        {
            'directed': False,
            'multigraph': False,
            'graph': {
                '__root__': 0,
                'label': 0,
                'accuracy': 0.9873,
                'precision': 0.9712,
                'recall': 1.0,
                'f1_score': 0.9854,
                'support': None,
                'f1_score-binary': 0.9854,
                'num_indices': 630,
                'feature_index': 8,
            },
            'nodes': [
                {'label': 1, '__root__': True, 'id': 0},
                {'label': 0, 'id': 1},
                {'label': 0, 'id': 2},
                {'label': 0, 'id': 3},
                {'label': 0, 'id': 4},
                {'label': 0, 'id': 5},
                {'label': 0, 'id': 6},
            ],
            'links': [
                {'label': 0, 'source': 0, 'target': 1},
                {'label': 0, 'source': 0, 'target': 3},
                {'label': 0, 'source': 0, 'target': 6},
                {'label': 0, 'source': 1, 'target': 2},
                {'label': 0, 'source': 1, 'target': 4},
                {'label': 0, 'source': 3, 'target': 4},
                {'label': 0, 'source': 3, 'target': 5},
            ],
        }
    )

    assert nx.is_isomorphic(G1, G2)
    assert not nx.is_isomorphic(G1, G3)

    assert gu.is_subgraph_isomorphic(G1, G2)
    assert gu.is_subgraph_isomorphic(G1, G3)


def test_merge_sets():
    G1 = nx.json_graph.node_link_graph(
        {
            'directed': False,
            'multigraph': False,
            'graph': {
                '__root__': 0,
                'f1_score': 0.9854,
                'name': 'G1',
            },
            'nodes': [
                {'__root__': True, 'label': 1, 'id': 0},
                {'label': 0, 'id': 1},
                {'label': 0, 'id': 3},
                {'label': 0, 'id': 6},
                {'label': 0, 'id': 2},
                {'label': 0, 'id': 4},
                {'label': 0, 'id': 5},
            ],
            'links': [
                {'source': 0, 'target': 1},
                {'source': 0, 'target': 3},
                {'source': 0, 'target': 6},
                {'source': 1, 'target': 2},
                {'source': 1, 'target': 4},
                {'source': 3, 'target': 4},
                {'source': 3, 'target': 5},
            ],
        }
    )

    G2 = nx.json_graph.node_link_graph(
        {
            'directed': False,
            'multigraph': False,
            'graph': {
                '__root__': 0,
                'f1_score': 0.9864,
                'name': 'G2',
            },
            'nodes': [
                {'__root__': True, 'label': 1, 'id': 0},
                {'label': 0, 'id': 1},
                {'label': 0, 'id': 3},
                {'label': 0, 'id': 6},
                {'label': 0, 'id': 7},
                {'label': 0, 'id': 2},
                {'label': 0, 'id': 4},
                {'label': 0, 'id': 5},
            ],
            'links': [
                {'source': 0, 'target': 1},
                {'source': 0, 'target': 3},
                {'source': 0, 'target': 6},
                {'source': 0, 'target': 7},
                {'source': 1, 'target': 2},
                {'source': 1, 'target': 4},
                {'source': 1, 'target': 5},
                {'source': 3, 'target': 2},
            ],
        }
    )

    G3 = nx.json_graph.node_link_graph(
        {
            'directed': False,
            'multigraph': False,
            'graph': {
                '__root__': 0,
                'f1_score': 0.9982,
                'name': 'G3',
            },
            'nodes': [
                {'__root__': True, 'label': 1, 'id': 0},
                {'label': 0, 'id': 1},
                {'label': 0, 'id': 3},
                {'label': 0, 'id': 7},
                {'label': 0, 'id': 2},
                {'label': 0, 'id': 4},
                {'label': 0, 'id': 5},
                {'label': 0, 'id': 6},
            ],
            'links': [
                {'source': 0, 'target': 1},
                {'source': 0, 'target': 3},
                {'source': 0, 'target': 7},
                {'source': 1, 'target': 2},
                {'source': 1, 'target': 4},
                {'source': 3, 'target': 4},
                {'source': 3, 'target': 5},
                {'source': 3, 'target': 6},
            ],
        }
    )

    G4 = nx.json_graph.node_link_graph(
        {
            'directed': False,
            'multigraph': False,
            'graph': {
                '__root__': 0,
                'f1_score': 0.9992,
                'name': 'G4',
            },
            'nodes': [
                {'__root__': True, 'label': 1, 'id': 0},
                {'label': 0, 'id': 1},
                {'label': 0, 'id': 4},
                {'label': 0, 'id': 2},
                {'label': 0, 'id': 3},
            ],
            'links': [
                {'source': 0, 'target': 1},
                {'source': 0, 'target': 4},
                {'source': 1, 'target': 4},
                {'source': 1, 'target': 2},
                {'source': 4, 'target': 3},
                {'source': 2, 'target': 3},
            ],
        }
    )

    H1 = nx.json_graph.node_link_graph(
        {
            'directed': False,
            'multigraph': False,
            'graph': {
                '__root__': 0,
                'f1_score': 0.9854,
                'name': 'H1',
            },
            'nodes': [
                {'__root__': True, 'label': 1, 'id': 0},
                {'label': 0, 'id': 1},
                {'label': 0, 'id': 3},
                {'label': 0, 'id': 6},
                {'label': 0, 'id': 2},
                {'label': 0, 'id': 4},
                {'label': 0, 'id': 5},
            ],
            'links': [
                {'source': 0, 'target': 1},
                {'source': 0, 'target': 3},
                {'source': 0, 'target': 6},
                {'source': 1, 'target': 2},
                {'source': 1, 'target': 4},
                {'source': 3, 'target': 4},
                {'source': 3, 'target': 5},
            ],
        }
    )

    H2 = nx.json_graph.node_link_graph(
        {
            'directed': False,
            'multigraph': False,
            'graph': {
                '__root__': 0,
                'f1_score': 0.9992,
                'name': 'H2',
            },
            'nodes': [
                {'__root__': True, 'label': 1, 'id': 0},
                {'label': 0, 'id': 1},
                {'label': 0, 'id': 4},
                {'label': 0, 'id': 2},
                {'label': 0, 'id': 3},
            ],
            'links': [
                {'source': 0, 'target': 1},
                {'source': 0, 'target': 4},
                {'source': 1, 'target': 4},
                {'source': 1, 'target': 2},
                {'source': 4, 'target': 3},
                {'source': 2, 'target': 3},
            ],
        }
    )

    H3 = nx.json_graph.node_link_graph(
        {
            'directed': False,
            'multigraph': False,
            'graph': {
                '__root__': 0,
                'f1_score': 0.5792,
                'name': 'H3',
            },
            'nodes': [
                {'label': 0, 'id': 3},
                {'label': 0, 'id': 1},
                {'label': 0, 'id': 2},
                {'__root__': True, 'label': 1, 'id': 0},
                {'label': 0, 'id': 6},
                {'label': 0, 'id': 4},
                {'label': 0, 'id': 5},
            ],
            'links': [
                {'source': 3, 'target': 1},
                {'source': 1, 'target': 2},
                {'source': 2, 'target': 0},
                {'source': 2, 'target': 6},
                {'source': 0, 'target': 6},
                {'source': 0, 'target': 4},
                {'source': 4, 'target': 5},
            ],
        }
    )

    merged = gu.MergeSets(
        [G1, G2, G3, G4],
        [H1, H2, H3],
        max_elements=4,
        sort_key='f1_score',
    ).merge()
    merged_name = [g.graph['name'] for g in merged]
    assert merged_name == ['H2', 'G3', 'G2', 'H3']


def test_num_unique_motifs(create_motif_test_graph):
    G = create_motif_test_graph()

    assert gu.num_unique_motifs(G, 3) == 2
    assert gu.num_unique_motifs(G, 4) == 5
    assert gu.num_unique_motifs(G, 5) == 12
    assert gu.num_unique_motifs(G, 6) == 37


def test_get_unique_motifs(create_motif_test_graph):
    G = create_motif_test_graph()

    print('num_nodes:', G.number_of_nodes(), G.number_of_edges())

    motifs_k3 = gu.get_unique_motifs(G, 3)
    assert len(motifs_k3) == 2

    m_k_4 = gu.get_unique_motifs(G, 4)
    assert len(m_k_4) == 5

    m_k_5 = gu.get_unique_motifs(G, 5)
    assert len(m_k_5) == 12

    m_k_6 = gu.get_unique_motifs(G, 6)
    assert len(m_k_6) == 37
