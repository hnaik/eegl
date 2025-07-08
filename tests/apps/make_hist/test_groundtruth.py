import unittest

import networkx as nx
from apps.make_hist.groundtruth import HouseMaker


class TestHouseMaker(unittest.TestCase):
    def test_make_house(self):
        hm = HouseMaker(700, 300)
        G = hm.make_house(300)
        assert G.number_of_nodes() == 5
        assert [int(n) for n in G.nodes] == [300, 301, 302, 303, 304]
        assert G.number_of_edges() == 6
        expected_edges = [
            (300, 301),
            (301, 302),
            (302, 303),
            (303, 300),
            (300, 304),
            (301, 304),
        ]

        for u, v in expected_edges:
            assert G.has_edge(u, v)

    def test_make_house_with_handle(self):
        hm = HouseMaker(700, 300)
        G = hm.make_house_with_handle(300, 0)
        assert G.number_of_nodes() == 6
        assert [int(n) for n in G.nodes] == [300, 301, 302, 303, 304, 0]
        assert G.number_of_edges() == 7
        expected_edges = [
            (300, 301),
            (301, 302),
            (302, 303),
            (303, 300),
            (300, 304),
            (301, 304),
            (300, 0),
        ]
        for u, v in expected_edges:
            assert G.has_edge(u, v)
