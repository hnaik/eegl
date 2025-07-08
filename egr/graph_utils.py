import logging
import math
import typing as ty

import igraph as ig
import networkx as nx

import egr.glasgow_subgraph_solver as gss

LOG = logging.getLogger(__name__)


def get_neighborhood_subgraph(G: nx.Graph, root_id: int, hops: int):
    return nx.ego_graph(G.copy(), root_id, hops, undirected=True)


def has_element(items: ty.Iterable[nx.Graph], G: nx.Graph) -> bool:
    assert isinstance(items, ty.Iterable), 'items is not an iterable'
    assert isinstance(G, nx.Graph), 'G is not of type nx.Graph'

    for H in items:
        if nx.is_isomorphic(G, H):
            return True
    return False


class MergeSets:
    def __init__(self, a, b, max_elements, sort_key):
        self._a = sorted(a, key=lambda x: x.graph[sort_key], reverse=True)
        self._b = sorted(b, key=lambda x: x.graph[sort_key], reverse=True)

        self.max_elements = max_elements
        self.sort_key = sort_key
        self.merged_set = []

    def merge(self):
        while len(self.merged_set) < self.max_elements:
            G = self._next_element()
            if G is None:
                break
            if not self._exists(G):
                self.merged_set.append(G)
        return self.merged_set

    def _next_element(self):
        if len(self._a) > 0 and len(self._b) > 0:
            if (
                self._a[0].graph[self.sort_key]
                > self._b[0].graph[self.sort_key]
            ):
                return self._a.pop(0)
            else:
                return self._b.pop(0)
        elif len(self._a) > 0:
            return self._a.pop(0)
        elif len(self._b) > 0:
            return self._b.pop(0)

    def _exists(self, G) -> bool:
        for H in self.merged_set:
            if is_subgraph_isomorphic(G, H):
                return True
        return False


def is_subgraph_isomorphic(G, H):
    if G.number_of_nodes() < H.number_of_nodes():
        return gss.subgraph_is_isomorphic(H, G)
    return gss.subgraph_is_isomorphic(G, H)


def num_unique_motifs(G: ig.Graph | nx.Graph, k: int) -> int:
    G = ig.Graph.from_networkx(G) if isinstance(G, nx.Graph) else G
    motifs = G.motifs_randesu(size=k)
    non_trivial = [m for m in motifs if not math.isnan(m) and m != 0]
    return len(non_trivial)


def get_unique_motifs(G: ig.Graph | nx.Graph, k: int) -> ty.List[nx.Graph]:
    G = ig.Graph.from_networkx(G) if isinstance(G, nx.Graph) else G
    proc = _IGraphMotifPostProcess()
    G.motifs_randesu(size=k, callback=proc)
    return proc.motifs


class _IGraphMotifPostProcess:
    def __init__(self):
        self.motifs = []
        self._classes = set([])

    def __call__(self, G: ig.Graph, nodes: ty.List, motif_class: int):
        H = G.induced_subgraph(nodes).copy().to_networkx()
        if motif_class in self._classes or not nx.is_connected(H):
            return
        self._classes.add(motif_class)
        self.motifs.append(H)
