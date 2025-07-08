from typing import List

import networkx as nx

from egr import graph_utils


def test_find_house_m0():
    G = nx.Graph()
    G.graph['random_count']: int = 300

    assert graph_utils.find_house_m0(G, node_id=300) == 300
    assert graph_utils.find_house_m0(G, node_id=301) == 300
    assert graph_utils.find_house_m0(G, node_id=302) == 300
    assert graph_utils.find_house_m0(G, node_id=303) == 300
    assert graph_utils.find_house_m0(G, node_id=304) == 300
    assert graph_utils.find_house_m0(G, node_id=305) == 305
    assert graph_utils.find_house_m0(G, node_id=309) == 305


def test_find_house_handles():
    G = nx.barabasi_albert_graph(300, m=5)
    G.graph['random_count']: int = 300
    G = graph_utils.attach_house(G, attach_node=0, m0=300)
    G.add_edge(1, 300)

    assert graph_utils.find_house_handles(G, node_id=300) == [0, 1]
