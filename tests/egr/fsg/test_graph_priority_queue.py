import networkx as nx

from egr.fsg import graph_priority_queue as gpq


def test_make_score_heap():
    label_graphs = {
        0: [nx.Graph(), nx.Graph(), nx.Graph()],
        1: [nx.Graph(), nx.Graph()],
    }

    label_graphs[0][0].graph['f1_score'] = 0.7
    label_graphs[0][0].graph['name'] = '0_0'
    label_graphs[0][0].add_edges_from([(0, 1), (1, 2), (2, 0)])

    label_graphs[0][1].graph['f1_score'] = 0.75
    label_graphs[0][1].graph['name'] = '0_1'
    label_graphs[0][1].add_edges_from([(0, 1), (1, 2), (2, 0)])

    label_graphs[0][2].graph['f1_score'] = 0.7
    label_graphs[0][2].graph['name'] = '0_2'
    label_graphs[0][2].add_edges_from([(0, 1), (1, 2)])

    label_graphs[1][0].graph['f1_score'] = 0.6
    label_graphs[1][0].graph['name'] = '1_0'
    label_graphs[1][0].add_edges_from([(0, 1), (1, 2), (2, 0)])

    label_graphs[1][1].graph['f1_score'] = 0.8
    label_graphs[1][1].graph['name'] = '1_1'
    label_graphs[1][1].add_edges_from([(0, 1), (1, 2), (2, 0)])

    assert label_graphs[0][0].graph['name'] == '0_0'
    assert label_graphs[0][1].graph['name'] == '0_1'
    assert label_graphs[0][2].graph['name'] == '0_2'
    assert label_graphs[1][0].graph['name'] == '1_0'
    assert label_graphs[1][1].graph['name'] == '1_1'

    score_heap = gpq.make_score_heap(label_graphs)

    pq_0 = score_heap[0]
    assert not pq_0.empty
    assert pq_0.pop().G.graph['name'] == '0_1'
    assert pq_0.pop().G.graph['name'] == '0_0'
    assert pq_0.pop().G.graph['name'] == '0_2'
    assert pq_0.empty

    pq_1 = score_heap[1]
    assert not pq_1.empty
    assert pq_1.pop().G.graph['name'] == '1_1'
    assert pq_1.pop().G.graph['name'] == '1_0'
    assert pq_1.empty
