import networkx as nx

import egr.fsg.frequent_pattern_finder as fpf

import egr

egr.init_logging()


def test_compute_score_matrix():
    G = nx.cycle_graph(6)
    num = G.number_of_nodes()
    G = nx.disjoint_union(G, nx.cycle_graph(6))
    G.add_edges_from([(i, i + num) for i in range(num)] + [(0, 7), (1, 8)])

    subgraphs = [nx.cycle_graph(3), nx.cycle_graph(4), nx.cycle_graph(5)]
    for H in subgraphs:
        H.graph['__root__'] = 0

    indices = [0, 2, 4, 6, 7, 9, 11]

    isomat = fpf.compute_iso_matrix(G, subgraphs, indices)
    print(isomat[:, 1][indices])
