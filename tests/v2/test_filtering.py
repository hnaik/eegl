import io
from types import SimpleNamespace
from unittest import mock

import networkx as nx

import egr.util as util
import egr.v2.filtering as flt


def test_get_previous_iter_candidates__iter_0():
    args = SimpleNamespace(iteration=0)
    assert flt.get_previous_iter_candidates(args) == []


def test_get_previous_iter_candidates():
    feature_dim = 10
    dir_path = mock.Mock()
    dir_path.glob.return_value = make_pathlike(feature_dim)

    args = SimpleNamespace(
        iteration=1,
        feature_dim=feature_dim,
        prev_fsg_dir=dir_path,
    )

    graphs, roots = flt.get_previous_iter_candidates(args)
    assert len(graphs) == args.feature_dim
    assert len(roots) == args.feature_dim

    for i in range(feature_dim):
        assert graphs[i].graph['__root__'] == roots[i]


def make_pathlike(num):
    def make_path(idx):
        path = mock.Mock()
        path.open.return_value = make_graph_data(idx)
        return path

    return [make_path(idx) for idx in range(num)]


def make_graph_data(root):
    G = nx.Graph()
    G.graph['__root__'] = root
    return io.StringIO(util.to_json(G))
