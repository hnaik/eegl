import logging
import typing as ty
from datetime import datetime

import networkx as nx
import torch

import egr.util as util
import egr.fsg.gaston as gaston
import egr.v2.iso_match as im

LOG = logging.getLogger(__name__)
FEATURE_DIM = 18

TargetGroupedGraphs = ty.Dict[str, ty.List[nx.Graph]]


def get_filtered_candidates(data, args):
    begin = datetime.now()

    unique_labels = torch.unique(data.y).tolist()
    expl_files = sorted(args.data_root.glob('*.json'))
    graphs = [util.load_graph(p) for p in expl_files]
    grouped = group_by_target_label(graphs, unique_labels)
    rooted_fsg = get_rooted_subgraphs(grouped, unique_labels, args)
    previous_rooted_fsg = get_previous_iter_candidates(args)

    computed_scores = {}
    for label in unique_labels:
        current = rooted_fsg[label]['fsg']
        previous = previous_rooted_fsg.get(int(label), [])
        if len(previous) > 0:
            current = im.remove_duplicates(current, previous, label)
        scores = im.compute_scores(data, label, current, True, 2)
        computed_scores.update({int(label): scores})

    candidates = pick_round_robin(
        computed_scores, unique_labels, feature_dim=FEATURE_DIM
    )
    assert (
        len(candidates) == FEATURE_DIM
    ), f'#candidates({len(candidates)}) != feature_dim({FEATURE_DIM})'
    return {
        'candidates': candidates,
        'timing': {'begin': begin, 'end': datetime.now()},
    }


def pick_round_robin(computed_scores, labels, feature_dim):
    LOG.info('picking round robin candidates')
    candidates = []
    index = 0
    label_indices = {label: 0 for label in labels}
    exhausted_count = 0
    while len(candidates) < feature_dim:
        if exhausted_count >= len(labels):
            break
        label = labels[index]
        scores = computed_scores[label]
        if label_indices[label] >= len(scores):
            LOG.info('exhausted label: %s', label)
            index += 1
            index %= len(labels)
            exhausted_count += 1
            continue

        candidate = scores[label_indices[label]]
        LOG.debug(
            'Adding candidate %d, label:%d, score:%f',
            len(candidates),
            label,
            candidate.graph['iso_match_score'],
        )
        candidates.append(candidate)
        label_indices[label] += 1
        index += 1
        index %= len(labels)

    return candidates


def get_rooted_subgraphs(grouped, labels, args) -> ty.Dict:
    LOG.info('Getting rooted subgraphs')
    subgraphs: ty.Dict = {label: {} for label in labels}
    for label, graphs in grouped.items():
        roots = [g.graph['__root__'] for g in graphs]
        fsg, fsg_roots = gaston.mineFreqRootedSubgraphs(
            graphs, roots, label, args, onlyMaximal=False
        )
        subgraphs[label].update({'fsg': fsg, 'roots': fsg_roots})
    return subgraphs


def group_by_target_label(
    graphs: ty.List[nx.Graph], target_classes: ty.List[str]
) -> TargetGroupedGraphs:
    grouped: TargetGroupedGraphs = {tclass: [] for tclass in target_classes}
    for G in graphs:
        grouped[G.graph['label']].append(G)
    return {
        k: sorted(v, key=lambda x: x.graph['__root__'])
        for k, v in grouped.items()
    }


def get_previous_iter_candidates(args) -> im.ScoreTable:
    # LOG.info('args=%s', args.__dict__.keys())
    if args.iteration == 0:
        return {}
    LOG.info('Getting previous candidates from: %s', args.prev_fsg_dir)
    LOG.info('prev_fsg_dir:%s', args.prev_fsg_dir)
    paths = args.prev_fsg_dir.glob('*.json')
    graphs = [util.load_graph(path) for path in paths]
    # roots = [g.graph['__root__'] for g in graphs]

    candidates: im.ScoreTable = {}
    for G in graphs:
        label = G.graph['label']
        if label not in candidates:
            candidates.update({label: []})
        candidates[label].append(G)
    return candidates
