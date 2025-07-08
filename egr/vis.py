import logging
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

LOG = logging.getLogger(__name__)


def draw_default(G: nx.Graph, **kwargs):
    labels = {n: n for n in G.nodes}
    if 'node_size' not in kwargs:
        kwargs.update({'node_size': 1000})
    if 'node_color' not in kwargs:
        kwargs.update({'node_color': '#bbb'})
    nx.draw(G, pos=nx.kamada_kawai_layout(G), labels=labels, **kwargs)


def make_labels(n: int) -> List[int]:
    return [n for n in range(1, n + 1)]


def make_heatmap(m: np.array, **kw):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_xticks(range(m.shape[1]))
    ax.set_xticklabels(make_labels(m.shape[1]))
    if 'aspect' not in kw:
        kw.update({'aspect': 'auto'})
    im = plt.imshow(m, **kw)
    fig.colorbar(im, ax=ax)
    kw = dict(rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(ax.get_xticklabels(), **kw)
    fig.tight_layout()
    plt.show()


def make_feature_heatmap(G: nx.Graph): ...


def show_fullerene_pattern(G, **kwargs):
    pos = nx.kamada_kawai_layout(G)
    edge_color = [G.edges[e]['color'] for e in G.edges]
    if 'node_size' not in kwargs:
        kwargs.update({'node_size': 400})
    if 'node_color' not in kwargs:
        kwargs.update({'node_color': '#bbddff'})
    if 'font_size' not in kwargs:
        kwargs.update({'font_size': 8})
    nx.draw(
        G,
        pos,
        with_labels=True,
        edge_color=edge_color,
        edge_cmap=plt.cm.Blues,
        **kwargs,
    )
    plt.show()
