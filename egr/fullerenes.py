import networkx as nx

from egr.graph import cycles as cyc


def label_fullerene(G: nx.Graph) -> nx.Graph:
    # H = nx.Graph()
    # H.add_nodes_from(G.nodes())
    # H.add_edges_from(G.edges())
    H = G.copy()

    for u, v in H.edges():
        H[u][v]['label'] = 0

    for n in H.nodes():
        cycles = cyc.find_bounded_cycles(H, n, 5)
        for cycle in cycles:
            Hs = nx.subgraph(H, cycle)
            for u, v in Hs.edges():
                H[u][v]['label'] = 1
    return H


def make_line_graph(G: nx.Graph):
    H = nx.line_graph(G)
    for u, v in H.nodes():
        H.nodes[(u, v)]['label'] = G[u][v]['label']

    return H
