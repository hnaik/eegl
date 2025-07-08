import logging

import networkx as nx

LOG = logging.getLogger(__name__)


def m7_add_edges(G: nx.Graph) -> nx.Graph:
    G.add_edges_from([(0, -1), (3, -1), (-1, -2)])
    return G


def m7_relabel(G: nx.Graph, start: int) -> nx.Graph:
    return nx.relabel_nodes(G, mapping={i: (i + start + 2) for i in G.nodes})


def motif_m7_a(start: int) -> nx.Graph:
    G: nx.Graph = nx.cycle_graph(6)
    H: nx.Graph = m7_relabel(m7_add_edges(G), start)
    labels = [1, 2, 3, 4, 4, 3, 4, 4]
    assert len(labels) == H.number_of_nodes()
    return H, labels


def motif_m7_b(start: int) -> nx.Graph:
    G1: nx.Graph = nx.cycle_graph(3)
    G2: nx.Graph = nx.cycle_graph(3)
    mapping = {i: (i + G1.number_of_nodes()) for i in G2.nodes()}
    G2 = nx.relabel_nodes(G2, mapping)
    H: nx.Graph = m7_relabel(m7_add_edges(nx.compose(G1, G2)), start)
    labels = [5, 6, 7, 8, 8, 7, 8, 8]
    assert len(labels) == H.number_of_nodes()
    return H, labels


def m8_add_edges(G: nx.Graph) -> nx.Graph:
    G.add_edges_from([(0, -1), (3, -1), (0, -2), (3, -2)])
    return G


def m8_relabel(G: nx.Graph, start: int) -> nx.Graph:
    return nx.relabel_nodes(G, mapping={i: (i + start + 2) for i in G.nodes})


def motif_m8_a(start: int) -> nx.Graph:
    G: nx.Graph = nx.cycle_graph(6)
    H: nx.Graph = m8_relabel(m8_add_edges(G), start)
    labels = [1, 2, 3, 4, 4, 3, 4, 4]
    assert len(labels) == H.number_of_nodes()
    return H, labels


def motif_m8_b(start: int) -> nx.Graph:
    G1: nx.Graph = nx.cycle_graph(3)
    G2: nx.Graph = nx.cycle_graph(3)
    mapping = {i: (i + G1.number_of_nodes()) for i in G2.nodes()}
    G2 = nx.relabel_nodes(G2, mapping)
    H: nx.Graph = m8_relabel(m8_add_edges(nx.compose(G1, G2)), start)
    labels = [5, 6, 7, 8, 8, 7, 8, 8]
    assert len(labels) == H.number_of_nodes()
    return H, labels


def reorder(G: nx.Graph) -> nx.Graph:
    H: nx.Graph = nx.Graph()
    H.add_nodes_from(sorted(G.nodes(data=True)))
    H.add_edges_from(G.edges(data=True))
    return H


def m9a_relabel(G: nx.Graph, start: int) -> nx.Graph:
    return nx.relabel_nodes(G, mapping={i: (i + start + 1) for i in G.nodes})


def motif_m9_a(start: int) -> nx.Graph:
    G: nx.Graph = nx.wheel_graph(7)
    G.add_edge(-1, 0)
    G: nx.Graph = m9a_relabel(G, start)
    H: nx.Graph = reorder(G)
    labels = [1, 2, 3, 3, 3, 3, 3, 3]
    assert len(labels) == H.number_of_nodes()
    return H, labels


def m9b_relabel(G: nx.Graph, start: int) -> nx.Graph:
    return nx.relabel_nodes(G, mapping={i: (i + start + 2) for i in G.nodes})


def m9b_add_edges(G: nx.Graph) -> nx.Graph:
    G.add_edges_from(
        [(0, -1), (3, -1), (-1, -2), (-1, 1), (-1, 2), (-1, 4), (-1, 5)]
    )
    return G


def motif_m9_b(start: int) -> nx.Graph:
    G1: nx.Graph = nx.cycle_graph(3)
    G2: nx.Graph = nx.cycle_graph(3)
    mapping = {i: (i + G1.number_of_nodes()) for i in G2.nodes()}
    G2 = nx.relabel_nodes(G2, mapping)
    G: nx.Graph = m9b_relabel(m9b_add_edges(nx.compose(G1, G2)), start)
    H: nx.Graph = reorder(G)
    labels = [4, 5, 6, 6, 6, 6, 6, 6]
    assert len(labels) == H.number_of_nodes()
    return H, labels


def motif_m10(
    num_vertices: int, num_parts: int, node_start: int, label_start: int
):
    G = nx.Graph()
    for _ in range(num_parts):
        G = nx.disjoint_union(G, nx.cycle_graph(num_vertices))

    G = nx.relabel_nodes(G, mapping={i: (i + node_start + 2) for i in G.nodes})
    labels = [label_start] * G.number_of_nodes()

    edges = [(n, node_start + 1) for n in G.nodes()]
    G.add_edges_from(edges)
    G.add_edge(node_start, node_start + 1)

    labels.extend([label_start + 1, label_start + 2])

    return G, labels
