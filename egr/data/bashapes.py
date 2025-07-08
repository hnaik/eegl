import logging

import networkx as nx

LOG = logging.getLogger(__name__)


class BAShapeGenerator:
    def __init__(self, **kw):
        self.__begin = 0 if 'begin' not in kw else kw['begin']
        self.__motif = dict(
            begin=300 if 'begin_motif' not in kw else kw['begin_motif'],
            end=700 if 'end_motif' not in kw else kw['end_motif'],
        )

    @staticmethod
    def build_graph(attach_node: int, anchor: int) -> nx.Graph:
        V = [v for v in range(attach_node, attach_node + 5)] + [anchor]
        G = nx.Graph()
        G.add_nodes_from(V)
        G.add_edges_from(
            [
                (attach_node + 0, attach_node + 1),
                (attach_node + 1, attach_node + 2),
                (attach_node + 2, attach_node + 3),
                (attach_node + 3, attach_node + 0),
                (attach_node + 0, attach_node + 4),
                (attach_node + 1, attach_node + 4),
                (attach_node + 0, anchor),
            ]
        )
        return G

    @staticmethod
    def find_anchors(G, attach_node):
        anchors = [n for n in G.neighbors(attach_node) if n < 300]
        if len(anchors) > 1:
            LOG.debug(
                'attach node %d has more than one anchor: %s',
                attach_node,
                ','.join([str(n) for n in anchors]),
            )
        return anchors

    def build_lookup(self, G):
        data = {}
        for node_id in range(self.__motif['begin'], self.__motif['end']):
            attach_node = node_id - (node_id % 5)
            if attach_node in data:
                data.update({node_id: data[attach_node]})
            else:
                anchors = self.find_anchors(G, attach_node)
                for anchor in anchors:
                    H: nx.Graph = self.build_graph(attach_node, anchor)
                    data.update({attach_node: H, anchor: H})
                    LOG.debug('Added anchor: %d', anchor)

        return data
