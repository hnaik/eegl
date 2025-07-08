import networkx as nx

import egr.v2.feature_generation as fg


def test_group_by_target_label():
    graphs = [
        nx.Graph(label=0, __root__=0),
        nx.Graph(label=1, __root__=1),
        nx.Graph(label=1, __root__=2),
        nx.Graph(label=0, __root__=3),
        nx.Graph(label=0, __root__=4),
        nx.Graph(label=1, __root__=5),
        nx.Graph(label=0, __root__=6),
        nx.Graph(label=0, __root__=7),
        nx.Graph(label=1, __root__=8),
    ]
    grouped = fg.group_by_target_label(graphs, ['0', '1'])

    assert len(grouped['0']) == 5
    assert len(grouped['1']) == 4

    assert [g.graph['__root__'] for g in grouped['0']] == [0, 3, 4, 6, 7]
    assert [g.graph['__root__'] for g in grouped['1']] == [1, 2, 5, 8]
