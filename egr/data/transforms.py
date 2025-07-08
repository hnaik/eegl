import logging
import typing as ty

import torch as th
import torch_geometric.data as pygdata

import apps.create_index as ci
from egr.datasets import noise

LOG = logging.getLogger(__name__)


def default_transform(data: pygdata.Data) -> pygdata.Data:
    return make_fold_masks(data, 10)


def airports_usa(data: pygdata.Data) -> pygdata.Data:
    return make_fold_masks(data, 10)


def citation_full_cora_ml(data: pygdata.Data) -> pygdata.Data:
    return make_fold_masks(data, 10)


def egr_logic(data: pygdata.Data) -> pygdata.Data:
    return make_fold_masks(data, 5)


def entities_mutag(data: pygdata.Data) -> pygdata.Data:
    LOG.debug('applying transforms to entities_mutag data %s', data)
    data.train_mask = th.full((data.num_nodes, 1), False, dtype=th.bool)
    data.train_mask[data.train_idx] = True
    data.val_mask = th.full((data.num_nodes, 1), False, dtype=th.bool)
    data.val_mask[data.test_idx] = True
    data.test_mask = th.full((data.num_nodes, 1), False, dtype=th.bool)
    data.test_mask[data.test_idx] = True

    data.y = th.zeros(data.num_nodes, dtype=th.long)
    data.y[data.train_idx] = data.train_y
    data.y[data.test_idx] = data.test_y

    data.x = make_vanilla_features(data, 32).x

    data.explain_idx = th.tensor(
        sorted(data.train_idx.tolist() + data.test_idx.tolist()), dtype=th.long
    )

    return data


def make_fold_masks(data: pygdata.Data, n_splits: int) -> pygdata.Data:
    splits = ci.make_folds(data.y.tolist(), n_splits)
    n = data.y.shape[0]
    data.train_mask = th.full((n, n_splits), False, dtype=th.bool)
    data.val_mask = th.full((n, n_splits), False, dtype=th.bool)
    data.test_mask = th.full((n, n_splits), False, dtype=th.bool)
    for fold, split in enumerate(splits):
        data.train_mask[split['train'], fold] = True
        data.val_mask[split['val'], fold] = True
        data.test_mask[split['test'], fold] = True
    return data


def make_vanilla_features(
    data: pygdata.Data, dim: ty.Optional[int] = None
) -> pygdata.Data:
    dim = dim or data.num_node_features
    data.x = th.ones((data.num_nodes, dim), dtype=th.float)
    return data
