import logging

import torch

import egr.fsg.feature_importance.threshold as thr

# import egr.fsg.frequent_pattern_finder as fpf
# from egr.fsg import gaston
# from egr.util import load_graph
# from egr.fsg import graph_priority_queue as gpq


LOG = logging.getLogger(__name__)


def apply_importance_filter(
    current_candidates, previous_candidates, labels, imp_scores, args
):
    LOG.info(
        'Applying feature importance using %d previous candidates',
        len(previous_candidates),
    )

    candidates = previous_candidates.copy()
    dim_labels = [p.graph['label'] for p in candidates]
    candidate_labels = torch.LongTensor(dim_labels)
    LOG.info('candidate_labels:%s', candidate_labels.tolist())
    indices = torch.arange(len(candidate_labels))
    labels = labels.to('cpu')
    unique_labels = torch.unique(labels)

    fi_params = args.feature_importance
    agg_strat, cfg = fi_params['aggregation'], fi_params['threshold']
    for label in unique_labels:
        label_scores = imp_scores[labels == label]
        LOG.info('Refining candidates for label:%d', label)
        agg_scores = compute_aggretate(label_scores, agg_strat)

        threshold = pick_threshold(
            agg_scores, label, dim_labels, cfg['type'], cfg['filter']
        )

        LOG.info(
            'Threshold for label %d: %s, scores:%s',
            label,
            threshold.item(),
            agg_scores,
        )

        label_indices = indices[candidate_labels == label]
        LOG.info('label_indices:%s', label_indices.tolist())
        for i in label_indices:
            if previous_candidates[i].graph['label'] != label:
                continue
            previous_f1 = previous_candidates[i].graph['f1_score']
            current_f1 = current_candidates[i].graph['f1_score']
            if agg_scores[i] < threshold and current_f1 > previous_f1:
                LOG.info('Replacing index %d with label %d', i, label)
                candidates[i] = current_candidates[i]
    return candidates


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


def pick_threshold(scores, label, labels, threshold_type, strategy) -> float:
    if threshold_type == 'single':
        return compute_threshold(scores, threshold_type, strategy)
    if threshold_type == 'multi':
        scores = [scores[i] for i, lab in enumerate(labels) if lab == label]
        scores = torch.FloatTensor(scores)
        return compute_threshold(scores, threshold_type, strategy)
    raise ValueError(f'Unknown threshold type: {threshold_type}')
