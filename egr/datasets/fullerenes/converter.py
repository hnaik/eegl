import logging
import time
from pathlib import Path

import networkx as nx
from joblib import Parallel, delayed
from networkx.algorithms import isomorphism
from tqdm import tqdm

import egr.datasets.fullerenes.motifs as motifs
import egr.glasgow_subgraph_solver as gss
import egr.util

LOG = logging.getLogger(__name__)
ROOT_URL = 'https://nanotube.msu.edu/fullerene/'


def make_graph_data(args, name, variant, matcher_type='nx') -> nx.Graph:
    matcher = get_matcher(matcher_type)
    mm = motifs.MotifMaker()
    H = make_fullerene_data(args, name=name, variant=variant)
    G = nx.line_graph(H)
    G.add_nodes_from((n, H.edges[n]) for n in G.nodes)
    G = nx.convert_node_labels_to_integers(G, label_attribute='original_label')

    pbar = tqdm(G.nodes, desc='Matching motifs')
    labels = [-1] * len(G.nodes)

    def annotate_node(node_id, node):
        g_copy = G.copy()
        g_copy.graph['root'] = node
        nx.set_node_attributes(g_copy, 0, '__root__')
        g_copy.nodes[node]['__root__'] = 1

        for motif_id, motif in enumerate(mm.line_patterns):
            if matcher(g_copy, motif.G):
                assert (
                    labels[node] == -1
                ), f'Node {node} matches multiple motifs, {labels[node]} and i'
                return node_id, motif_id

    begin = time.time()
    result = Parallel(n_jobs=-1)(
        delayed(annotate_node)(node_id, node)
        for node_id, node in enumerate(pbar)
    )

    for node_id, motif_id in result:
        pbar.set_description(f'Labeling node {node_id}/{len(G.nodes)}')
        labels[node_id] = motif_id
    end = time.time()

    unique_labels = sorted(set(labels))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    normalized_labels = [label_map[label] for label in labels]

    for n in G.nodes:
        G.nodes[n]['label'] = normalized_labels[n]
        G.nodes[n]['y'] = normalized_labels[n]

    LOG.debug('Labels: %s', normalized_labels)
    LOG.info('Time taken: %.2f seconds', end - begin)

    return G


def node_match(node1, node2):
    return node1['__root__'] == node2['__root__']


def subgraph_is_isomorphic(G, H):
    GM = isomorphism.GraphMatcher(G, H, node_match=node_match)
    return GM.subgraph_is_isomorphic()


def get_matcher(matcher_type):
    if matcher_type == 'gss':
        return gss.subgraph_is_isomorphic
    elif matcher_type == 'nx':
        return subgraph_is_isomorphic
    raise ValueError(f'Unknown matcher type: {matcher_type}')


def make_fullerene_data(args, name, variant):
    path = get_src_file_path(args, name, variant)
    edges = []
    for line in path.open():
        bits = line.split()
        if len(bits) != 8:
            continue
        nodes = [int(x) for x in bits[-4:]]
        edges.extend([(nodes[0], n) for n in nodes[1:]])
    G = nx.Graph()
    G.add_edges_from(edges)
    return G


def get_src_file_path(args, name, variant):
    path = Path(args.temp_dir) / f'{name}/{variant}.xyz'
    if not path.exists():
        url = f'{ROOT_URL}{name}/{variant}.xyz'
        LOG.info('Downloading %s from %s', path, url)
        egr.util.download(url, path)
    return path
