import logging
from pathlib import Path
import typing as ty

import torch
import torch_geometric.data as pgd
from torch_geometric.utils import convert

from apps.create_index import make_folds
from egr.data.io import EgrDenseData
from egr.util import DATASET_DIR
from egr.datasets import noise

LOG = logging.getLogger(__name__)


class EgrBaseDataset(pgd.InMemoryDataset):
    def __init__(self, cfg: ty.Dict, name: str, *args, **kwargs):
        self._cfg = cfg
        self.name = name
        LOG.debug('args=%s, kw=%s', args, kwargs)
        super().__init__(*args, **kwargs)
        self.load(self.processed_paths[0])

    @property
    def has_val_idx(self) -> bool:
        return True

    @property
    def raw_file_names(self):
        return self._raw_file_names()

    def _raw_file_names(self):
        return ['raw_data.pt']

    @property
    def processed_file_names(self):
        return self._processed_file_names()

    def _processed_file_names(self):
        return ['processed.pt']

    @property
    def label_path(self) -> Path:
        return DATASET_DIR / 'input_data' / self.name / 'labels.txt'

    @property
    def graph_path(self) -> Path:
        return DATASET_DIR / 'input_data' / self.name / self.graph_filename

    @property
    def graph_filename(self) -> str:
        return '0001.json'

    def download(self) -> None: ...

    def process(self):
        LOG.debug('processing data.name=%s', self.name)
        egr_data = self.load_data()
        N = egr_data.G.number_of_nodes()
        data = convert.from_networkx(egr_data.G)
        data.x = self.generate_init_features(N, self.dim)
        data.y = egr_data.y

        data.train_mask = torch.zeros(N, self.folds, dtype=torch.bool)
        data.val_mask = torch.zeros(N, self.folds, dtype=torch.bool)
        data.test_mask = torch.zeros(N, self.folds, dtype=torch.bool)

        folds = make_folds(
            egr_data.y.tolist(), self.folds, val_idx=self.has_val_idx
        )

        for i, fold in enumerate(folds):
            data.train_mask[:, i][fold['train']] = True
            data.test_mask[:, i][fold['test']] = True

            if self.has_val_idx:
                data.val_mask[:, i][fold['val']] = True

        noise_cfg = self._cfg.get('noise', {})
        LOG.info('noise_cfg: %s', noise_cfg)

        for noise_type, fraction in noise_cfg.items():
            noise_fn = getattr(noise, f'make_{noise_type}_noise')
            LOG.info('Applying noise %s fraction %s', noise_type, fraction)
            noise_fn(data, fraction)

        self.save([data], self.processed_paths[0])

    def load_data(self):
        return self._load_data()

    def _load_data(self):
        return EgrDenseData.read_new(self.graph_path, self.label_path)

    def generate_init_features(self, N: int, F: int) -> torch.Tensor:
        LOG.info('generating features of same color N=%s, F=%s', N, F)
        return torch.ones(N, F)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__.lower()}_{self.name}'
