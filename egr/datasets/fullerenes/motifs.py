import logging
import typing as ty
from types import SimpleNamespace

import networkx as nx

LOG = logging.getLogger(__name__)


def make_pattern(cycles, colored_edges=None):
    G = nx.Graph()
    for cycle in cycles:
        nx.add_cycle(G, cycle)

    nx.set_edge_attributes(G, 0, 'label')
    nx.set_edge_attributes(G, 'black', 'color')
    colored_edges = colored_edges or {(0, 1): 'red'}
    if colored_edges:
        for (i, j), color in colored_edges.items():
            G.edges[(i, j)]['color'] = color
            G.edges[(i, j)]['label'] = 1
    return G


class MotifMaker:
    @property
    def hh_pp(self) -> ty.List[ty.List[int]]:
        return make_pattern(
            [
                [0, 1, 2, 3, 4, 5],
                [0, 1, 6, 7, 8, 9],
                [1, 2, 10, 11, 6],
                [0, 5, 12, 13, 9],
            ]
        )

    @property
    def hh_ph(self) -> ty.List[ty.List[int]]:
        return make_pattern(
            [
                [0, 1, 2, 3, 4, 5],
                [0, 1, 6, 7, 8, 9],
                [1, 2, 10, 11, 6],
                [0, 5, 12, 13, 14, 9],
            ]
        )

    @property
    def hh_hh(self) -> ty.List[ty.List[int]]:
        return make_pattern(
            [
                [0, 1, 2, 3, 4, 5],
                [0, 1, 6, 7, 8, 9],
                [1, 2, 10, 11, 12, 6],
                [0, 5, 13, 14, 15, 9],
            ]
        )

    @property
    def ph_pp(self) -> ty.List[ty.List[int]]:
        return make_pattern(
            [
                [0, 1, 2, 3, 4],
                [0, 1, 5, 6, 7, 8],
                [1, 2, 9, 10, 5],
                [0, 4, 11, 12, 8],
            ]
        )

    @property
    def ph_ph(self) -> ty.List[ty.List[int]]:
        return make_pattern(
            [
                [0, 1, 2, 3, 4],
                [0, 1, 5, 6, 7, 8],
                [1, 2, 9, 10, 5],
                [0, 4, 11, 12, 13, 8],
            ]
        )

    @property
    def ph_hh(self) -> ty.List[ty.List[int]]:
        return make_pattern(
            [
                [0, 1, 2, 3, 4],
                [0, 1, 5, 6, 7, 8],
                [1, 2, 9, 10, 11, 5],
                [0, 4, 12, 13, 14, 8],
            ]
        )

    @property
    def pp_pp(self) -> ty.List[ty.List[int]]:
        return make_pattern(
            [
                [0, 1, 2, 3, 4],
                [0, 1, 5, 6, 7],
                [1, 2, 8, 9, 5],
                [0, 4, 10, 11, 7],
            ]
        )

    @property
    def pp_ph(self) -> ty.List[ty.List[int]]:
        return make_pattern(
            [
                [0, 1, 2, 3, 4],
                [0, 1, 5, 6, 7],
                [1, 2, 8, 9, 5],
                [0, 4, 10, 11, 12, 7],
            ]
        )

    @property
    def pp_hh(self) -> ty.List[ty.List[int]]:
        return make_pattern(
            [
                [0, 1, 2, 3, 4],
                [0, 1, 5, 6, 7],
                [1, 2, 8, 9, 10, 5],
                [0, 4, 11, 12, 13, 7],
            ]
        )

    @property
    def patterns(self) -> ty.List[ty.List[int]]:
        return [
            {'g': self.hh_pp, 'name': 'Hexagon-Hexagon-Pentagon-Pentagon'},
            {'g': self.hh_ph, 'name': 'Hexagon-Hexagon-Pentagon-Hexagon'},
            {'g': self.hh_hh, 'name': 'Hexagon-Hexagon-Hexagon-Hexagon'},
            {'g': self.ph_pp, 'name': 'Pentagon-Hexagon-Pentagon-Pentagon'},
            {'g': self.ph_ph, 'name': 'Pentagon-Hexagon-Pentagon-Hexagon'},
            {'g': self.ph_hh, 'name': 'Pentagon-Hexagon-Hexagon-Hexagon'},
            {'g': self.pp_pp, 'name': 'Pentagon-Pentagon-Pentagon-Pentagon'},
            {'g': self.pp_ph, 'name': 'Pentagon-Pentagon-Pentagon-Hexagon'},
            {'g': self.pp_hh, 'name': 'Pentagon-Pentagon-Hexagon-Hexagon'},
        ]

    @property
    def line_patterns(self) -> ty.List[ty.List[int]]:
        return [self.convert_to_linegraph(p) for p in self.patterns]

    def convert_to_linegraph(self, p) -> SimpleNamespace:
        g = nx.line_graph(p['g'])
        g.add_nodes_from((node, p['g'].edges[node]) for node in g.nodes)
        g = nx.convert_node_labels_to_integers(g, label_attribute='orig')
        nx.set_node_attributes(g, 0, '__root__')
        roots = []
        for node in g.nodes:
            if g.nodes[node]['label'] == 1:
                g.nodes[node]['color'] = '#ffbbbb'
                g.nodes[node]['__root__'] = 1
                roots.append(node)
            else:
                g.nodes[node]['color'] = '#bbddff'
        assert len(roots) == 1, f'Expected one root, found {len(roots)}'
        g.graph['__root__'] = roots[0]
        return SimpleNamespace(G=g, name=p['name'], root=roots[0])
