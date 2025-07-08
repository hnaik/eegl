import logging
from typing import Iterable

import networkx as nx

LOG = logging.getLogger(__name__)


def find_house_m0(G: nx.Graph, node_id: int):
    random_count: int = G.graph['random_count']
    assert node_id >= random_count, f'{node_id} < {random_count}'
    offset: int = (node_id - G.graph['random_count']) % 5
    return node_id - offset


def find_house_handles(G: nx.Graph, node_id: int):
    m0: int = find_house_m0(G, node_id)
    return [int(n) for n in G.neighbors(m0) if int(n) < m0]


def attach_house(G: nx.Graph, attach_node: int, m0: int) -> nx.Graph:
    assert G.has_node(attach_node)
    G.add_edges_from(
        [
            (attach_node, m0),
            (m0, m0 + 1),
            (m0 + 1, m0 + 2),
            (m0 + 2, m0 + 3),
            (m0 + 3, m0),
            (m0, m0 + 4),
            (m0 + 1, m0 + 4),
        ]
    )
    return G


def label_fullerene(G: nx.Graph) -> nx.Graph:
    node_cycles = {}
    for n in list(G.nodes()):
        cycles = nx.cycle_basis(G, n)
        cycle_classes = {0: [], 1: []}
        for i, cycle in enumerate(cycles):
            size = len(cycle)
            if size == 5:
                cycle_classes[0].append(cycle)
            elif size == 6:
                cycle_classes[1].append(cycle)
                node_cycles.update({n: cycle_classes})

    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(G.edges())

    attrs = {}
    for u, v in G.edges():
        for label, cycles in node_cycles[u].items():
            for cycle in cycles:
                if v in cycle:
                    print(f'({u}, {v}) = {label}')
                    attrs.update({(u, v): {'label': label}})
        if attrs.get((u, v)) is None:
            for label, cycles in node_cycles[v].items():
                for cycle in cycles:
                    if u in cycle:
                        print(f'({u}, {v}) = {label}')
                        attrs.update({(u, v): {'label': label}})
        if (u, v) not in attrs:
            attrs.update({(u, v): {'label': 2}})
    nx.set_edge_attributes(H, attrs)
    return H
