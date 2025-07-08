import json
import logging
import signal
import typing as ty
from collections import defaultdict
from datetime import datetime
from multiprocessing.pool import AsyncResult
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Tuple

import networkx as nx
import torch
from torch import Tensor

import egr.fsg.feature_importance.threshold as thr
import egr.fsg.frequent_pattern_finder as fpf
from egr.fsg import gaston
from egr.fsg import graph_priority_queue as gpq
from egr.fsg.feature_importance import apply_importance_filter
from egr.util import load_graph
import egr.graph_utils as gu

LOG = logging.getLogger(__name__)


def read_json(data_dir: Path) -> Iterable[ty.Dict]:
    LOG.debug('reading path %s', data_dir)
    for path in sorted(data_dir.glob('*.json')):
        LOG.debug('loading %s', path)
        yield json.load(path.open())


def create_graph(graph_data):
    nodes_data = graph_data['nodes']
    G = nx.Graph()
    root = -1
    id_to_node = {}
    for i, n_d in enumerate(nodes_data):
        id = n_d['id']
        id_to_node[id] = i
        G.add_node(i)
        for k, v in n_d.items():
            G.nodes[i][k] = v
        if '__root__' in n_d and n_d['__root__'] == 1:
            root = i
            G.graph['explanation_for_node'] = n_d['original']
            G.graph['explained_label'] = n_d['label']

    edges_data = graph_data['links']
    for e_d in edges_data:
        s, t = e_d['source'], e_d['target']
        G.add_edge(id_to_node[s], id_to_node[t], attr=e_d)
    return G, root


def load_subgraphs(root_dir: Path) -> Tuple[List, List]:
    LOG.debug('load_subgraphs() %s', root_dir)
    graphs, roots = [], []
    for data in read_json(root_dir.expanduser()):
        G, root = create_graph(data)
        if root == -1:
            continue
        gaston.makeRootNode(G, root)
        graphs.append(G)
        roots.append(root)
    return graphs, roots


def make_partitions(graphs: List[nx.Graph], roots: List[int]) -> ty.Dict:
    assert len(graphs) == len(roots)
    partitions: ty.Dict = {}
    for i, G in enumerate(graphs):
        label = G.graph['explained_label']
        if label not in partitions:
            partitions.update({label: {'graphs': [], 'roots': []}})
        partitions[label]['graphs'].append(G)
        partitions[label]['roots'].append(roots[i])
    return partitions


def indices_for_labels(indices: torch.Tensor, labels: Tensor) -> ty.Dict:
    data: ty.Dict = {}
    label_array = labels.tolist()
    for idx in sorted(indices.tolist()):
        label = label_array[idx]
        if label not in data:
            data.update({label: []})
        data[label].append(idx)
    return data


Results = List[AsyncResult]

POOL_ARGS = dict(
    initializer=signal.signal, initargs=(signal.SIGINT, signal.SIG_IGN)
)


def get_prev_iter_fsg(dirpath: Path) -> ty.Dict:
    prev_map = defaultdict(list)
    if dirpath is None:
        return prev_map
    LOG.info('Getting previous iteration patterns from %s', dirpath)

    graphs = [load_graph(p) for p in dirpath.glob('*.json')]
    graphs = sorted(graphs, key=lambda x: x.graph['feature_index'])
    for G in graphs:
        prev_map[G.graph['label']].append({'graph': G, **G.graph})
    return prev_map


def order_scores(scores: List, key: str = 'f1_score') -> List:
    return sorted(scores, key=lambda x: x.graph[key], reverse=True)


def stringify_scores(s_list: List) -> str:
    return ','.join([f'{s["f1_score"]:.3f}' for s in s_list])


def filter_graphs(data, args) -> ty.Dict[int, List[nx.Graph]]:
    LOG.info('Filtering graphs in %s', args.data_root)
    graphs, roots = load_subgraphs(args.data_root)
    labels = data.y.tolist()
    partitions = make_partitions(graphs, roots)
    prev_scores = get_prev_iter_fsg(args.prev_fsg_dir)

    LOG.debug('loaded previous scores')
    for label, v in prev_scores.items():
        f1scores = ['{:.3f}'.format(item['f1_score']) for item in v]
        LOG.debug('label:%d, f1-score:%s', label, ','.join(f1scores))

    all_subgraphs = []
    LOG.info(
        'Computing frequent subgraphs for %d labels, maximal_only: %s',
        len(partitions),
        args.maximal_only,
    )
    for label, partition in sorted(partitions.items()):
        subgraphs, roots = gaston.mineFreqRootedSubgraphs(
            graphs=partition['graphs'],
            roots=partition['roots'],
            label=label,
            onlyMaximal=args.maximal_only,
            args=args,
        )
        all_subgraphs.append((label, subgraphs))

    train_indices = (
        torch.nonzero(data.train_mask, as_tuple=False).squeeze(1).tolist()
    )
    feature_dim = data.x.shape[1]
    with_scores = {}
    for label, subgraphs in all_subgraphs:
        if not subgraphs or len(subgraphs) == 0:
            LOG.error('No subgraphs found for label %s', label)
            continue

        prev_label_scores = [g['graph'] for g in prev_scores[label]]

        LOG.info(
            'Computing F1-Scores for label %d from %d subgraphs, %d previous',
            label,
            len(subgraphs),
            len(prev_label_scores),
        )
        begin_label = datetime.now()
        curr_label_scores = fpf.compute_scores(
            data.G, subgraphs, train_indices, label, labels, args
        )
        ordered_current_scores = order_scores(
            [g['graph'] for g in curr_label_scores]
        )
        dur_label = datetime.now() - begin_label

        num_scores = min(len(curr_label_scores), feature_dim)
        curr_label_scores = ordered_current_scores

        merger = gu.MergeSets(
            prev_label_scores,
            curr_label_scores,
            max_elements=feature_dim,
            sort_key='f1_score',
        )
        sorted_scores = [{'graph': g, **g.graph} for g in merger.merge()]

        LOG.info(
            'Label=%d(current:%02d,prev:%02d), dur:%s, scores(top-%d):%s',
            label,
            len(curr_label_scores),
            len(prev_label_scores),
            dur_label,
            num_scores,
            stringify_scores(sorted_scores[:num_scores]),
        )
        with_scores.update({label: sorted_scores})
    return with_scores


def format_metrics(metrics: ty.Dict) -> str:
    m = SimpleNamespace(**metrics)
    return (
        f'A:{m.accuracy:.3f}, P:{m.precision:.3f}, R:{m.recall:.3f}, '
        f'F1:{m.f1_score:.3f}'
    )


def compute_aggretate(scores, strategy: str) -> torch.Tensor:
    agg_method = getattr(torch, strategy)
    agg = agg_method(scores, dim=0)
    return agg


def compute_threshold(scores, threshold_type: str, strategy: str) -> float:
    threshold = getattr(thr, strategy)(scores)
    LOG.info(
        'Threshold type:%s, strategy:%s, value:%s',
        threshold_type,
        strategy,
        threshold.item(),
    )
    return threshold


def read_previous_fsg(dirpath: Path) -> ty.List:
    return [load_graph(p) for p in dirpath.glob('*.json')]


def get_data_dim(data, args) -> int:
    if isinstance(args.data_dim, int):
        return args.data_dim
    if args.data_dim == 'auto':
        return data.x.shape[1]
    raise ValueError(f'Invalid data_dim: {args.data_dim}')


def filter_graphs_with_feature_importance(data, args) -> ty.List[nx.Graph]:
    data_dim = get_data_dim(data, args)
    LOG.info('Filtering graphs with feature importance, data_dim:%d', data_dim)
    filtered_graphs = filter_graphs(data, args)
    current = pick_patterns_round_robin(filtered_graphs, data_dim)

    if not hasattr(args, 'feature_importance'):
        LOG.info('Not applying feature importance, feature_importance not set')
        return current

    feature_importance = args.feature_importance
    if feature_importance['min_iteration'] > args.iteration:
        LOG.info(
            'Not applying feature importance, '
            'feature_importance.min_iteration=%s, current iteration:%s',
            feature_importance['min_iteration'],
            args.iteration,
        )
        return current

    LOG.info('Reading previous fsg graphs from %s', args.prev_fsg_dir)
    x_importance = torch.load(args.data_root / 'feature_importance.pt')
    previous = [load_graph(p) for p in args.prev_fsg_dir.glob('*.json')]
    previous = sorted(previous, key=lambda g: g.graph['feature_index'])
    return apply_importance_filter(
        current, previous, data.y, x_importance, args
    )


def pick_patterns_round_robin(data, data_dim) -> List[nx.Graph]:
    graphs = []
    labeled_pqs = gpq.make_score_heap(data, 'f1_score')
    labels = sorted(data.keys())
    max_dim = min(sum([len(s) for _, s in data.items()]), data_dim)

    while len(graphs) < max_dim:
        for label in labels:
            pq = labeled_pqs[label]
            if pq.empty:
                continue
            item = pq.pop()
            graphs.append(item.G.copy())
            if len(graphs) >= max_dim:
                break
    LOG.info(
        'data_dim:%d, max_dim:%d, pattern labels=%s',
        data_dim,
        max_dim,
        [G.graph['label'] for G in graphs],
    )
    return graphs
