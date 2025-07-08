from __future__ import annotations

import json
import logging
import pickle
import typing as ty
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import torch
from torch_geometric import datasets
import torch_geometric.data as pygdata
from torch import Tensor

import egr.util as eu
from egr.util import load_features, load_graph, load_labels

LOG = logging.getLogger(__name__)


class EgrData:
    def __str__(self) -> str:
        return f'|N|={self.N},|E|={self.num_edges}'

    @property
    def num_nodes(self) -> int:
        return self.V.numel()

    @property
    def N(self) -> int:
        return self.num_nodes

    @property
    def num_edges(self) -> int:
        return self.E.size(0)

    @property
    def directed(self) -> bool:
        return self.G.is_directed()


@dataclass
class EgrDenseData(EgrData):
    y: Tensor
    G: nx.Graph
    X: ty.Optional[Tensor] = None

    @classmethod
    def read(cls, path: Path) -> EgrDenseData:
        G: nx.Graph = load_graph(path)
        LOG.debug('Loaded %s from %s path', G, path)
        y = Tensor(G.graph['labels']).type(torch.LongTensor)
        for n in G.nodes:
            G.nodes[n]['feat']: List = Tensor(G.nodes[n]['feat'])
        return cls(y, G=G)

    @classmethod
    def read_new(
        cls,
        graph_path: Path,
        label_path: Path,
        feature_path: Optional[Path] = None,
        feature_importance: Optional[np.ndarray] = None,
    ) -> EgrDenseData:
        G: nx.Graph = cls.load_graph(graph_path)
        y = load_labels(label_path.open())
        assert G.number_of_nodes() == y.shape[0]

        if feature_path is not None:
            LOG.info(
                'loading features from: %s, %s',
                feature_path,
                G.number_of_nodes(),
            )
            h = load_features(feature_path)
            # LOG.info('H=%s', h[:10])
            if feature_importance is not None:
                LOG.info('Using feature importance')
                h += feature_importance
            # LOG.info('H=%s', h[:10])
            nx.set_node_attributes(G, {i: {'feat': h[i]} for i in G.nodes()})
            X = np.float64(h)
            X = torch.from_numpy(X)
            return cls(y, G=G, X=X)
        return cls(y, G=G)

    @staticmethod
    def load_graph(path: Path) -> nx.Graph:
        G = nx.Graph()
        data = json.load(path.open())
        u, v = data['u'], data['v']
        assert len(u) == len(v)
        G.add_nodes_from([n for n in range(data['n'])])
        G.add_edges_from([(u[i], v[i]) for i in range(len(u))])
        return G

    @classmethod
    def load_pyg(
        cls,
        graph_path: Path,
        label_path: Path,
        feature_path: Path,
        index_path: Path,
    ) -> pygdata.Data:
        data = cls.read_new(graph_path, label_path, feature_path)
        edges = [e for e in data.G.edges()] + [(v, u) for u, v in data.G.edges]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        indices = json.load(index_path.open())
        # train_index = torch.tensor(sorted(indices['train']), dtype=torch.long)
        # mask = [False] * data.G.number_of_nodes()
        # train_mask = torch.tensor(mask, dtype=torch.bool)
        # train_mask[train_index] = True
        n: int = data.G.number_of_nodes()
        train_index, train_mask = get_mask(sorted(indices['train']), n)
        val_index, val_mask = get_mask(sorted(indices['val']), n)
        test_index, test_mask = get_mask(sorted(indices['test']), n)

        num_classes = data.y.unique().size()[0]

        return pygdata.Data(
            G=data.G,
            x=data.X.clone().detach().float(),
            edge_index=edge_index,
            y=data.y.long(),
            train_index=train_index,
            train_mask=train_mask,
            val_index=val_index,
            val_mask=val_mask,
            test_index=test_index,
            test_mask=test_mask,
            num_classes=num_classes,
        )

    @staticmethod
    def to_compact(G: nx.Graph) -> Dict:
        u, v = list(zip(*[e for e in G.edges()]))
        return dict(n=G.number_of_nodes(), u=u, v=v)

    @classmethod
    def save(cls, G: nx.Graph, path: Path):
        data = cls.to_compact(G)
        json.dump(data, fp=path.open('w'))

    @classmethod
    def convert_pickled(
        cls, src_path: Path, dst_dir: Path, sample_id: ty.Optional[str] = None
    ) -> nx.Graph:
        dst_dir.mkdir(parents=True, exist_ok=True)
        sample_id = sample_id or '0001'
        dst_path: Path = dst_dir / f'{sample_id}.json'
        with src_path.open('rb') as f:
            G: nx.Graph = pickle.load(f)
            LOG.info('Saving graph to %s', dst_path)
            cls.save(G, dst_path)

            data = eu.to_dict(G)
            y = ','.join([str(n.get('y')) for n in data['nodes']])
            label_path = dst_dir / 'labels.txt'
            LOG.info('Saving labels to %s', label_path)
            label_path.open('w').write(y)

    @classmethod
    def from_pickled(cls, src_path: Path) -> nx.Graph:
        with src_path.open('rb') as f:
            G = pickle.load(f)
            data = eu.to_dict(G)
            y = torch.LongTensor([n.get('y') for n in data['nodes']])
            unique_labels = torch.unique(y).tolist()
            LOG.info(
                'Loading from pickled file %s, N=%d, #C=%d',
                src_path,
                y.shape[0],
                len(unique_labels),
            )
            return cls(y, G=G)


def get_mask(indices: ty.List, n: int) -> ty.Tuple[torch.Tensor, torch.Tensor]:
    index = torch.tensor(indices, dtype=torch.long)
    mask = torch.tensor([False] * n, dtype=torch.bool)
    mask[index] = True
    return index, mask
