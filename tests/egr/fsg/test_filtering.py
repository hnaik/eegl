import networkx as nx
import numpy as np
import torch
from types import SimpleNamespace

import egr
from egr.fsg import filtering as flt

egr.init_logging()


def test_apply_importance_filter():
    labels = torch.LongTensor([0, 1, 2, 0, 1, 2, 0, 1, 2])

    imp_arr = []
    for i in range(len(labels)):
        a = np.roll(np.arange(9, step=1.25) * 0.1, i * 2)
        imp_arr.append(a.tolist())
    imp_scores = torch.FloatTensor(imp_arr)

    previous_candidates = []
    current_candidates = []
    for i in range(imp_scores.shape[1]):
        label = i % 3
        g_previous = nx.Graph()
        g_previous.graph.update({'label': label, 'round': 1, 'f1_score': 0.7})
        previous_candidates.append(g_previous)

        g_current = nx.Graph()
        g_current.graph.update({'label': label, 'round': 2, 'f1_score': 0.8})
        current_candidates.append(g_current)

    assert len(previous_candidates) == len(current_candidates)

    args = SimpleNamespace(
        feature_importance={
            'aggregation': 'mean',
            'threshold': {'type': 'multi', 'filter': 'mean'},
        }
    )

    candidates = flt.apply_importance_filter(
        current_candidates,
        previous_candidates,
        labels,
        imp_scores,
        args,
    )

    assert len(candidates) == len(previous_candidates)

    # assert candidates[0].graph['round'] == 2
    # assert candidates[1].graph['round'] == 1
    # assert candidates[2].graph['round'] == 2
    # assert candidates[3].graph['round'] == 1
    # assert candidates[4].graph['round'] == 1
    # assert candidates[5].graph['round'] == 2
    # assert candidates[6].graph['round'] == 2
    # assert candidates[7].graph['round'] == 1


def test_pick_patterns_round_robin():
    label_graphs = {
        0: [
            {'graph': nx.Graph()},
            {'graph': nx.Graph()},
            {'graph': nx.Graph()},
        ],
        1: [{'graph': nx.Graph()}, {'graph': nx.Graph()}],
    }

    label_graphs[0][0]['graph'].graph['f1_score'] = 0.7
    label_graphs[0][0]['graph'].graph['name'] = '0_0'
    label_graphs[0][0]['graph'].graph['label'] = 0
    label_graphs[0][0]['graph'].add_edges_from([(0, 1), (1, 2), (2, 0)])

    label_graphs[0][1]['graph'].graph['f1_score'] = 0.75
    label_graphs[0][1]['graph'].graph['name'] = '0_1'
    label_graphs[0][1]['graph'].graph['label'] = 0
    label_graphs[0][1]['graph'].add_edges_from([(0, 1), (1, 2), (2, 0)])

    label_graphs[0][2]['graph'].graph['f1_score'] = 0.7
    label_graphs[0][2]['graph'].graph['name'] = '0_2'
    label_graphs[0][2]['graph'].graph['label'] = 0
    label_graphs[0][2]['graph'].add_edges_from([(0, 1), (1, 2)])

    label_graphs[1][0]['graph'].graph['f1_score'] = 0.6
    label_graphs[1][0]['graph'].graph['name'] = '1_0'
    label_graphs[1][0]['graph'].graph['label'] = 1
    label_graphs[1][0]['graph'].add_edges_from([(0, 1), (1, 2), (2, 0)])

    label_graphs[1][1]['graph'].graph['f1_score'] = 0.8
    label_graphs[1][1]['graph'].graph['name'] = '1_1'
    label_graphs[1][1]['graph'].graph['label'] = 1
    label_graphs[1][1]['graph'].add_edges_from([(0, 1), (1, 2), (2, 0)])

    patterns = flt.pick_patterns_round_robin(label_graphs, 5)

    assert patterns[0].graph['name'] == '0_1'
    assert patterns[1].graph['name'] == '1_1'
    assert patterns[2].graph['name'] == '0_0'
    assert patterns[3].graph['name'] == '1_0'
    assert patterns[4].graph['name'] == '0_2'


def test_pick_patterns_round_robin__duplicates():
    label_graphs = {
        0: [
            {'graph': nx.Graph()},
            {'graph': nx.Graph()},
            {'graph': nx.Graph()},
        ],
        1: [{'graph': nx.Graph()}, {'graph': nx.Graph()}],
    }

    label_graphs[0][0]['graph'].graph['f1_score'] = 0.7
    label_graphs[0][0]['graph'].graph['name'] = '0_0'
    label_graphs[0][0]['graph'].graph['label'] = 0
    label_graphs[0][0]['graph'].add_edges_from([(0, 1), (1, 2), (2, 0)])

    label_graphs[0][1]['graph'].graph['f1_score'] = 0.75
    label_graphs[0][1]['graph'].graph['name'] = '0_1'
    label_graphs[0][1]['graph'].graph['label'] = 0
    label_graphs[0][1]['graph'].add_edges_from([(0, 1), (1, 2), (2, 0)])

    label_graphs[0][2]['graph'].graph['f1_score'] = 0.7
    label_graphs[0][2]['graph'].graph['name'] = '0_2'
    label_graphs[0][2]['graph'].graph['label'] = 0
    label_graphs[0][2]['graph'].add_edges_from([(0, 1), (1, 2)])

    label_graphs[1][0]['graph'].graph['f1_score'] = 0.6
    label_graphs[1][0]['graph'].graph['name'] = '1_0'
    label_graphs[1][0]['graph'].graph['label'] = 1
    label_graphs[1][0]['graph'].add_edges_from([(0, 1), (1, 2), (2, 0)])

    label_graphs[1][1]['graph'].graph['f1_score'] = 0.8
    label_graphs[1][1]['graph'].graph['name'] = '1_1'
    label_graphs[1][1]['graph'].graph['label'] = 1
    label_graphs[1][1]['graph'].add_edges_from([(0, 1), (1, 2), (2, 0)])

    patterns = flt.pick_patterns_round_robin(label_graphs, 5)

    assert patterns[0].graph['name'] == '0_1'
    assert patterns[1].graph['name'] == '1_1'
    assert patterns[2].graph['name'] == '0_0'
    assert patterns[3].graph['name'] == '1_0'
    assert patterns[4].graph['name'] == '0_2'
