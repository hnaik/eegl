import logging
import random
import typing as ty

import networkx as nx
import numpy as np
import torch
import torch_geometric.utils as pygutils

LOG = logging.getLogger(__name__)


def add_noise(data: nx.Graph, cfg: ty.Dict) -> nx.Graph:
    LOG.info('Adding noise to %s', data)

    num_nodes_to_scramble = int(data.num_nodes * cfg['fraction']) + 1
    LOG.info(
        'Adding noise type:%s to %d/%d (%.2f%%) nodes',
        cfg['type'],
        num_nodes_to_scramble,
        data.num_nodes,
        cfg['fraction'] * 100,
    )
    nodes = list(range(data.num_nodes))
    scramble = np.random.choice(nodes, size=num_nodes_to_scramble)
    LOG.info('orig nodes: %s', nodes)
    LOG.info('to scramble: %s', scramble)
    LOG.info('before scramble labels=%s', data.y[scramble])
    unique_labels = torch.unique(data.y).tolist()
    for idx in scramble:
        orig = data.y[idx]
        labels = unique_labels.copy()
        labels.remove(orig)
        label = np.random.choice(labels, size=1)
        LOG.info('%3d: %d -> %d', idx, data.y[idx], label[0])
        data.y[idx] = int(label[0])
    LOG.info('after scramble labels=%s', data.y[scramble])
    return data


def add_noise_to_folds(data: nx.Graph, cfg: ty.Dict):
    LOG.info('Adding noise to folds')
    data.labels = data.zeros(data.train_mask.shape)
    n_folds: int = data.train_mask.shape[1]
    for fold in range(n_folds):
        data.labels[:, fold] = add_noise_to_fold(data, cfg, fold)
    return data


def add_noise_to_fold(data, fraction, fold):
    LOG.debug('Adding noise to fold %d', fold)
    train_indices = data.train_mask[:, fold].nonzero().squeeze(1)
    num_indices = len(train_indices)
    num_nodes_to_scramble = round(num_indices * fraction)
    LOG.debug(
        'Adding noise to %d/%d (%.2f%%) nodes',
        num_nodes_to_scramble,
        num_indices,
        fraction * 100,
    )
    scramble = np.random.choice(train_indices, size=num_nodes_to_scramble)
    LOG.debug('train indices: %s', train_indices.tolist())
    LOG.debug('to scramble: %s', scramble.tolist())
    LOG.info('fold:%d bef. scramble=%s', fold, data.y[scramble].tolist())
    unique_labels = torch.unique(data.y).tolist()
    y_copy = data.y.clone()
    for idx in scramble:
        labels = unique_labels.copy()
        labels.remove(data.y[idx])
        label = random.choice(labels)
        y_copy[idx] = label
        LOG.debug('%3d: %d -> %d', idx, data.y[idx], y_copy[idx])
    LOG.info('fold:%d aft. scramble=%s', fold, y_copy[scramble].tolist())
    return y_copy


def shuffle_labels(y, indices, fraction):
    num_shuffle = round(len(indices) * fraction)
    shuffled_indices = np.random.choice(indices, size=num_shuffle)
    LOG.info('bfr=%s', y[shuffled_indices].tolist())
    LOG.debug('shuffle=%s', shuffled_indices.tolist())
    unique_labels = torch.unique(y).tolist()
    for idx in shuffled_indices:
        labels = unique_labels.copy()
        labels.remove(y[idx])
        label = random.choice(labels)
        y[idx] = label
    return shuffled_indices


def make_label_noise(data, fraction):
    LOG.info('Adding label noise')
    n_folds = data.train_mask.shape[1]
    data.y_noise = torch.stack((data.y,) * n_folds).t()
    for fold in range(n_folds):
        indices = data.train_mask[:, fold].nonzero().squeeze(1)
        shuffled_indices = shuffle_labels(
            data.y_noise[:, fold], indices, fraction
        )
        LOG.info(
            'Label noise applied: %s',
            data.y[shuffled_indices].tolist(),
        )
    return data


def make_edge_addition_noise(data, fraction):
    G = pygutils.to_networkx(data, to_undirected=True)
    num_edges_to_add = int(G.number_of_edges() * fraction)
    LOG.info(
        'Adding %d/%d (%.2f%%) edges',
        num_edges_to_add,
        G.number_of_edges(),
        fraction * 100,
    )

    edges_added = 0
    while edges_added < num_edges_to_add:
        u = random.choice(list(G.nodes))
        v = random.choice(list(G.nodes))
        if u == v or G.has_edge(u, v):
            continue
        G.add_edge(u, v)
        edges_added += 1

    data_temp = pygutils.from_networkx(G)
    data.edge_index = data_temp.edge_index
    return data


def make_edge_removal_noise(data, fraction):
    G = pygutils.to_networkx(data, to_undirected=True)
    num_edges_to_remove = int(G.number_of_edges() * fraction)
    LOG.info(
        'Removing %d/%d (%.2f%%) edges',
        num_edges_to_remove,
        G.number_of_edges(),
        fraction * 100,
    )

    edges_removed = 0
    while edges_removed < num_edges_to_remove:
        u, v = random.choice(list(G.edges))
        if G.has_edge(u, v) and G.degree[u] > 1 and G.degree[v] > 1:
            G.remove_edge(u, v)
            edges_removed += 1
    data_temp = pygutils.from_networkx(G)
    data.edge_index = data_temp.edge_index
    return data
