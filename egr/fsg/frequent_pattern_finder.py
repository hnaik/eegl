import logging
import platform
import tempfile
import typing as ty
from pathlib import Path

import networkx as nx
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    f1_score,
)
from tqdm import tqdm
from joblib import Parallel, delayed

import egr.graph_utils as gu
import egr.glasgow_subgraph_solver as gss

LOG = logging.getLogger(__name__)


def compute_iso_matrix(
    G: nx.Graph, subgraphs: ty.List[nx.Graph], indices: ty.Collection
) -> np.array:
    iso_matrix: np.array = np.zeros([G.number_of_nodes(), len(subgraphs)])

    hops = max([nx.eccentricity(H, H.graph['__root__']) for H in subgraphs])

    kw = {} if platform.system() == 'Darwin' else dict(prefix='/dev/shm/')
    with tempfile.TemporaryDirectory(**kw) as name:
        target_dir = Path(name) / 'target_dir'
        pattern_dir = Path(name) / 'pattern_dir'

        target_dir.mkdir(parents=True, exist_ok=True)
        pattern_dir.mkdir(parents=True, exist_ok=True)

        target_files = {i: gen_target(G, hops, target_dir, i) for i in indices}
        pattern_files = [
            gen_pattern(H, pattern_dir, i) for i, H in enumerate(subgraphs)
        ]
        params = []
        for r, target_path in target_files.items():
            for c, pattern_path in enumerate(pattern_files):
                params.append((r, c, pattern_path, target_path))

        pbar = tqdm(params)
        result = Parallel(n_jobs=-2)(delayed(is_iso)(*args) for args in pbar)
        for i, j, iso in result:
            iso_matrix[i, j] = iso
    return iso_matrix


def is_iso(
    row: int, col: int, pattern_path: Path, target_path: Path
) -> ty.Tuple[int, np.array]:
    return row, col, gss.solve(pattern_path, target_path)


def gen_target(G, hops, target_dir, node_id):
    csv_path = target_dir / f'{node_id:02}.csv'
    try:
        H = gu.get_neighborhood_subgraph(G, node_id, hops)
        nx.set_node_attributes(H, 0, '__root__')
        H.nodes[node_id]['__root__'] = 1
        H.graph['__root__'] = node_id
        gss.save_csv(H, csv_path)
    except nx.exception.NetworkXError as e:
        LOG.error('Error in gen_target: %s, G:%s, node_id:%s', e, G, node_id)
        LOG.error('G.nodes: %s', G.nodes)
        raise
    return csv_path


def gen_pattern(H: nx.Graph, pattern_dir: Path, index: int) -> Path:
    H.nodes[H.graph['__root__']]['__root__'] = True
    csv_path = pattern_dir / f'{index:02}.csv'
    gss.save_csv(H, csv_path)
    return csv_path


def compute_scores(
    G: nx.Graph,
    subgraphs: ty.List[nx.Graph],
    indices: ty.List,
    target_label: int,
    labels: ty.List[int],
    args,
):
    iso: np.ndarray = compute_iso_matrix(G, subgraphs, indices)
    true_labels = [target_label == labels[i] for i in indices]
    scores: ty.List[ty.Dict] = []

    for i, H in enumerate(subgraphs):
        pred_labels = iso[:, i][indices]
        precision, recall, fscore, support = precision_recall_fscore_support(
            true_labels,
            pred_labels,
            average=args.average_strategy,
            zero_division=0,
            pos_label=True,
        )
        metrics = {
            'accuracy': accuracy_score(true_labels, pred_labels),
            'precision': precision,
            'recall': recall,
            'f1_score': fscore,
            'support': support,
            'f1_score-binary': f1_score(
                true_labels, pred_labels, pos_label=True
            ),
            'num_indices': len(indices),
        }
        H.graph['label'] = target_label
        H.graph.update(**metrics)

        scores.append({'graph': H, **metrics})

    return scores
