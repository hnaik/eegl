"""
```shell
$ python apps/make_new_variants.py \
    --data-dir=/data/results/input_data \
    --input-variant=f10 \
    --output-variant=f13
```
"""

import logging
import typing as ty
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, List, Tuple

import networkx as nx
import numpy as np

from apps.gnn_explainer.gengraph import perturb
from egr.data import variants
from egr.data.variants import motif_m7_a, motif_m7_b
from egr.data.io import EgrDenseData
from egr.log import init_logging

LOG = logging.getLogger('make_new_variants')


HOUSE_RANGE_BEGIN = 300
HOUSE_RANGE_END = 700
EXTRA_RANGE_BEGIN = HOUSE_RANGE_END
HOUSE_STRIDE = 5


def add_edges(G: nx.Graph, edges: List):
    H: nx.Graph = G.copy()
    H.add_edges_from(edges)
    return H


def remove_edges(G: nx.Graph, edges: List):
    H: nx.Graph = G.copy()
    H.remove_edges_from(edges)
    return H


def make_m2(G: nx.Graph) -> nx.Graph:
    return remove_edges(G, [(n + 1, n + 4) for n in range(300, 700, 5)])


def make_m3(G: nx.Graph) -> nx.Graph:
    return add_edges(make_m2(G), [(n, n + 2) for n in range(300, 700, 5)])


def make_m4(G: nx.Graph) -> nx.Graph:
    return add_edges(make_m2(G), [(n + 1, n + 3) for n in range(300, 700, 5)])


def make_m5(G: nx.Graph) -> nx.Graph:
    edges = []
    extra_node = EXTRA_RANGE_BEGIN
    for n in range(HOUSE_RANGE_BEGIN, HOUSE_RANGE_END, HOUSE_STRIDE):
        edges.extend([(n, extra_node), (n + 2, extra_node)])
        extra_node += 1
    return add_edges(G, edges)


def make_m6(G: nx.Graph) -> nx.Graph:
    edges = []
    extra_node = EXTRA_RANGE_BEGIN
    for n in range(HOUSE_RANGE_BEGIN, HOUSE_RANGE_END, HOUSE_STRIDE):
        edges.extend([(n + 1, extra_node), (n + 3, extra_node)])
        extra_node += 1
    return add_edges(G, edges)


def make_m7(G_unused: nx.Graph) -> nx.Graph:
    num_motifs = 160
    G: nx.Graph = nx.barabasi_albert_graph(n=num_motifs * 3.75, m=5)
    N: int = G.number_of_nodes()
    spacing = N // num_motifs
    attach_nodes = [int(k * spacing) for k in range(num_motifs)]
    all_labels = [0] * N
    for i in range(num_motifs):
        start: int = G.number_of_nodes()
        H, labels = motif_m7_a(start) if i % 2 == 0 else motif_m7_b(start)
        all_labels.extend(labels)
        G = nx.union(G, H)
        G.add_edge(start, attach_nodes[i])
    G = perturb([G], 0.01)[0]
    return G, all_labels


def make_p007_02(*args) -> nx.Graph:
    num_motifs: int = 80
    G: nx.Graph = nx.barabasi_albert_graph(n=300, m=5)
    N: int = G.number_of_nodes()
    spacing = N // num_motifs
    attach_nodes: List[int] = [int(k * spacing) for k in range(num_motifs)]
    all_labels: List[int] = [0] * N
    for i in range(num_motifs):
        start: int = G.number_of_nodes()
        H, labels = motif_m7_a(start) if i % 2 == 0 else motif_m7_b(start)
        all_labels.extend(labels)
        G: nx.Graph = nx.union(G, H)
        G.add_edge(start, attach_nodes[i])
    G = perturb([G], 0.01)[0]
    return G, all_labels


def make_m8(G_unused: nx.Graph) -> nx.Graph:
    num_motifs = 160
    G: nx.Graph = nx.barabasi_albert_graph(n=num_motifs * 3.75, m=5)
    N: int = G.number_of_nodes()
    spacing = N // num_motifs
    attach_nodes = [int(k * spacing) for k in range(num_motifs)]
    all_labels = [0] * N
    for i in range(num_motifs):
        start: int = G.number_of_nodes()
        H, labels = (
            variants.motif_m8_a(start)
            if i % 2 == 0
            else variants.motif_m8_b(start)
        )
        all_labels.extend(labels)
        G = nx.union(G, H)
        G.add_edge(start, attach_nodes[i])
    G = perturb([G], 0.01)[0]
    return G, all_labels


def make_p008_02(
    n: int = 300, m: int = 5, num_motifs: int = 80, *args
) -> nx.Graph:
    G: nx.Graph = nx.barabasi_albert_graph(n=n, m=m)
    N: int = G.number_of_nodes()
    spacing = N // num_motifs
    attach_nodes = [int(k * spacing) for k in range(num_motifs)]
    all_labels = [0] * N
    for i in range(num_motifs):
        start: int = G.number_of_nodes()
        H, labels = (
            variants.motif_m8_a(start)
            if i % 2 == 0
            else variants.motif_m8_b(start)
        )
        all_labels.extend(labels)
        G = nx.union(G, H)
        G.add_edge(start, attach_nodes[i])
    G = perturb([G], 0.01)[0]
    return G, all_labels


def mk_p008_02_a(
    n: int = 300, m: int = 5, num_motifs: int = 80, *args
) -> nx.Graph:
    G: nx.Graph = nx.barabasi_albert_graph(n=n, m=m)
    N: int = G.number_of_nodes()
    spacing = N // num_motifs
    attach_nodes = [int(k * spacing) for k in range(num_motifs)]
    all_labels = [0] * N
    for i in range(num_motifs):
        start: int = G.number_of_nodes()
        H, labels = variants.motif_m8_a(start)
        all_labels.extend(labels)
        G = nx.union(G, H)
        G.add_edge(start, attach_nodes[i])
    G = perturb([G], 0.01)[0]
    return G, all_labels


def make_m9(G_unused: nx.Graph) -> nx.Graph:
    num_motifs = 160
    G: nx.Graph = nx.barabasi_albert_graph(n=num_motifs * 3.75, m=5)
    N: int = G.number_of_nodes()
    spacing = N // num_motifs
    attach_nodes = [int(k * spacing) for k in range(num_motifs)]
    all_labels = [0] * N
    for i in range(num_motifs):
        start: int = G.number_of_nodes()
        H, labels = (
            variants.motif_m9_a(start)
            if i % 2 == 0
            else variants.motif_m9_b(start)
        )
        all_labels.extend(labels)
        G = nx.union(G, H)
        G.add_edge(start, attach_nodes[i])
    G = perturb([G], 0.01)[0]
    return G, all_labels


def make_p009_02(G_unused: nx.Graph) -> nx.Graph:
    num_motifs = 80
    G: nx.Graph = nx.barabasi_albert_graph(n=300, m=5)
    N: int = G.number_of_nodes()
    spacing = N // num_motifs
    attach_nodes = [int(k * spacing) for k in range(num_motifs)]
    all_labels = [0] * N
    for i in range(num_motifs):
        start: int = G.number_of_nodes()
        H, labels = (
            variants.motif_m9_a(start)
            if i % 2 == 0
            else variants.motif_m9_b(start)
        )
        all_labels.extend(labels)
        G = nx.union(G, H)
        G.add_edge(start, attach_nodes[i])
    G = perturb([G], 0.01)[0]
    return G, all_labels


def make_m10(_) -> nx.Graph:
    num_motifs: int = 150
    G: nx.Graph = nx.barabasi_albert_graph(n=num_motifs * 3.75, m=5)
    N: int = G.number_of_nodes()
    spacing = N // num_motifs
    param_list = [
        {'num_vertices': 3, 'num_parts': 4, 'label_start': 1},
        {'num_vertices': 4, 'num_parts': 3, 'label_start': 4},
        {'num_vertices': 6, 'num_parts': 2, 'label_start': 7},
    ]
    spacing = N // num_motifs
    all_labels = [0] * N
    for i in range(num_motifs):
        param_index = i % 3
        params = param_list[param_index]
        start: int = G.number_of_nodes()
        params.update({'node_start': start})
        H, labels = variants.motif_m10(**params)
        all_labels.extend(labels)
        G = nx.union(G, H)
        attach_node = i * spacing
        G.add_edge(start, attach_node)
    G = perturb([G], 0.01)[0]
    return G, all_labels


def make_m11(G: nx.Graph) -> nx.Graph:
    labels: np.array = G.graph['labels']
    edges = set([])
    for u in range(300):
        N = nx.neighbors(G, u)
        middle_neighbors = [n for n in N if labels[n] == 1]
        if len(middle_neighbors) == 0:
            continue
        for m in middle_neighbors:
            same = [m_ for m_ in nx.neighbors(G, m) if labels[m_] == 1]
            v = same[0]
            if len(same) > 1:
                assert (m + 1) in same
                v = m + 1
            if not G.has_edge(u, v) and not G.has_edge(v, u):
                edges.add((u, v))

    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(G.edges())
    H.add_edges_from(edges)
    LOG.info('G(%s) -> H(%s) %d', G, H, len(H.edges))
    LOG.info('adding %d edges %s', len(edges), edges)
    H.add_edges_from(edges)
    LOG.info('G(%s) -> H(%s) %d', G, H, len(H.edges))
    return H, labels


def make_p001_04(G: nx.Graph) -> nx.Graph:
    labels: np.array = G.graph['labels']
    edges = set([])
    for u in range(300):
        if labels[u] != 4:
            continue
        N = nx.neighbors(G, u)
        middle_neighbors = [n for n in N if labels[n] == 1]
        if len(middle_neighbors) == 0:
            continue
        for m in middle_neighbors:
            same = [m_ for m_ in nx.neighbors(G, m) if labels[m_] == 1]
            v = same[0]
            if len(same) > 1:
                assert (m + 1) in same
                v = m + 1
            if not G.has_edge(u, v) and not G.has_edge(v, u):
                edges.add((u, v))

    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(G.edges())
    H.add_edges_from(edges)
    LOG.debug('G(%s) -> H(%s) %d', G, H, len(H.edges))
    LOG.debug('adding %d edges %s', len(edges), edges)
    H.add_edges_from(edges)
    LOG.debug('G(%s) -> H(%s) %d', G, H, len(H.edges))
    return H, labels


def make_p001_05(G: nx.Graph) -> ty.Tuple[nx.Graph, ty.List]:
    """variant p001-05

    $ python apps/mk_orbit_variants.py \
        --data-dir=/data/results/input_data \
        --input-variant=p001-02 \
        --output-variant=p001-05

    """
    labels: np.array = G.graph['labels']
    e_remove = set([])
    e_add = set([])
    for u in range(300):
        if labels[u] != 4:
            continue
        neighbors: ty.List = [v for v in nx.neighbors(G, u) if labels[v] == 1]
        for v in neighbors:
            e_remove.add((u, v))
            for w in nx.neighbors(G, v):
                if labels[w] == 3:
                    e_add.add((u, w))
    G = remove_edges(G, e_remove)
    G = add_edges(G, e_add)
    return G, labels


def main(args):
    LOG.info('Starting app with log-level %s', args.log_level)
    input_dir: Path = args.data_dir / args.input_variant
    output_dir: Path = args.data_dir / args.output_variant
    output_dir.mkdir(parents=True, exist_ok=True)

    func_suffix = args.output_variant.replace('-', '_')
    func: Callable = globals()[f'make_{func_suffix}']
    LOG.info('Running function %s()', func.__name__)

    labels = None
    LOG.info('Input dir %s', input_dir)
    label_path: Path = input_dir / 'labels.txt'
    samples = [f'{sid + 1:04d}.json' for sid in range(args.num_samples)]
    for sample_file_name in samples:
        input_path = input_dir / sample_file_name
        LOG.debug('Reading input %s', input_path)
        data: EgrDenseData = EgrDenseData.read_new(input_path, label_path)
        G = data.G
        # G.graph['labels'] = data.y.tolist()
        H, labels = func(G)
        G.graph['labels'] = labels
        output_path: Path = output_dir / input_path.name
        LOG.info(
            'Reading %s(|N|=%d,|E|=%d), writing to %s(|N|=%d,|E|=%d)',
            input_path,
            G.number_of_nodes(),
            G.number_of_edges(),
            output_path,
            H.number_of_nodes(),
            H.number_of_edges(),
        )
        EgrDenseData.save(H, output_path)
        string_labels = [str(l) for l in G.graph['labels']]
        labels_path: Path = output_dir / 'labels.txt'
        LOG.info('Saving labels file to %s', labels_path)
        labels_path.open('w').write(','.join(string_labels))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', '--data-dir', type=Path, required=True)
    parser.add_argument('-i', '--input-variant', type=str, required=True)
    parser.add_argument(
        '-o',
        '--output-variant',
        type=str,
        required=True,
        choices=[
            'p001-03',
            'p001-04',
            'p001-05',
            'p007-02',
            'p008-02',
            'p009-02',
        ],
    )
    parser.add_argument('--num-samples', type=int, default=5)
    parser.add_argument(
        '-l',
        '--log-level',
        type=str,
        default='info',
        choices=['debug', 'info', 'warning', 'error', 'critical'],
    )
    args = parser.parse_args()
    init_logging(args.log_level)
    main(args)
