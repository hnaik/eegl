import logging
import typing as ty

import networkx as nx
import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn import metrics
from tqdm import tqdm

import egr.glasgow_subgraph_solver as gss

LOG = logging.getLogger(__name__)

ScoreTable: ty.TypeAlias = ty.Dict[str, ty.List[nx.Graph]]


def compute_scores(data, label: int, subgraphs, sort, beta=2) -> ty.List:
    scores: ty.List = []

    train_indices = data.train_index.tolist()
    train_y = data.y[data.train_index]
    y_true = train_y == label
    LOG.info('computing scores for label=%s', label)
    matches = compute_match_matrix(data.G, train_indices, subgraphs)

    for sg_idx, sg in enumerate(subgraphs):
        y_match = matches[:, sg_idx]
        score = metrics.fbeta_score(y_true, y_match, beta=beta)
        H = sg.copy()
        H.graph['iso_match_score'] = score
        scores.append(H)

    unique_scores = set([g.graph['iso_match_score'] for g in scores])
    LOG.info('unique_scores:%s', unique_scores)
    if sort:
        scores = sorted(scores, key=lambda x: x.graph['iso_match_score'])
    return scores


def compute_match_matrix(
    G: nx.Graph, nodes: ty.List, subgraphs: ty.List[nx.Graph]
) -> np.ndarray:
    rows = len(nodes)
    cols = len(subgraphs)
    m = torch.zeros((rows, cols), dtype=bool)
    LOG.debug(
        'Performing %dx%d=%d computations, into matrix=%s',
        rows,
        cols,
        rows * cols,
        m.shape,
    )
    params = []
    for node_idx, node in enumerate(nodes):
        for sg_idx, sg in enumerate(subgraphs):
            params.append((G, sg, node, node_idx, sg_idx))

    pbar = tqdm(params)
    parallel = Parallel(n_jobs=-2, return_as='generator')
    result = parallel(delayed(is_subgraph_match)(*args) for args in pbar)

    for r, c, iso in result:
        m[r, c] = iso

    return m


def is_subgraph_match(G, H, node, row, col) -> ty.Tuple[int, int, bool]:
    # root_id = H.graph['__root__']
    # hops = nx.eccentricity(H, root_id)
    target = G.copy()
    target.nodes[node]['__root__'] = True
    # target = nx.ego_graph(G, node, hops, undirected=True)
    return row, col, gss.subgraph_is_isomorphic(target, H)


# def compute_match_row(G, subgraphs, node, idx):
#     row = torch.zeros(len(subgraphs), dtype=bool)
#     for idx, h in tqdm(enumerate(subgraphs)):
#         hops = nx.eccentricity(h, h.graph['__root__'])
#         g = nx.ego_graph(G, node, hops, undirected=True)
#         row[idx] = gss.subgraph_is_isomorphic(g, h)
#     return idx, row


def remove_duplicates(current, previous, label):
    filtered = []

    LOG.debug('current: %d, previous: %d', len(current), len(previous))
    bar = tqdm(current)
    n_duplicates = 0
    for G in bar:
        duplicate = False
        for H in previous:
            if is_isomorphic(G, H):
                duplicate = True
                n_duplicates += 1
                bar.set_description(f'label:{label} {n_duplicates} dropped')
                break
        if not duplicate:
            filtered.append(G)

    LOG.debug('retained %d/%d', len(filtered), len(current))
    return filtered


def is_isomorphic(G1: nx.Graph, G2: nx.Graph) -> bool:
    def node_match(n1, n2) -> bool:
        has_root = '__root__' in n1 and '__root__' in n2
        return has_root and n1['__root__'] == n2['__root__']

    return nx.is_isomorphic(G1, G2, node_match=node_match)


# def compute_scores(data, rooted_fsg) -> ScoreTable:
#     scores: ScoreTable = {}
#     for label, subgraphs in rooted_fsg.items():
#         scores.update({label: []})

#         train_y = data.y[data.train_idx]
#         y_true = train_y == label
#         LOG.info('computing scores for label=%s', label)
#         matches = compute_match_matrix(data.G, data.train_idx, subgraphs)

#         for sg_idx, sg in enumerate(subgraphs):
#             y_match = matches[:, sg_idx]
#             score = f1_score(
#                 y_true,
#                 y_match,
#                 pos_label=True,
#                 average='weighted',
#                 zero_division=0,
#             )

#             H = sg.copy()
#             H.graph['iso_match_score'] = score
#             scores[label].append(H)

#     return scores
