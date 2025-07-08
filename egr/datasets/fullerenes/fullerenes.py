import logging

import torch

from egr.datasets.egr_base_dataset import EgrBaseDataset
from egr.data.io import EgrDenseData
from egr.util import DATASET_DIR

LOG = logging.getLogger(__name__)


DEFAULT_DIMS = {
    'g180': 56,
    'g180-v2': 56,
    'c60': 18,
    'c70': 18,
}


class Fullerenes(EgrBaseDataset):
    without_val = set(['g180', 'g180-v2'])

    @property
    def graph_path(self) -> str:
        return DATASET_DIR / f'pickled/fullerenes/{self.name}.pkl'

    @property
    def has_val_idx(self) -> bool:
        return self.name.lower() not in self.without_val

    @property
    def dim(self) -> int:
        if self.init_features['dim'] > 0:
            LOG.info('Using spec feature_dim=%s', self.init_features['dim'])
            return self.init_features['dim']
        default_dims = DEFAULT_DIMS.get(self.name.lower(), 18)
        LOG.info('Default feature_dim=%d for %s', default_dims, self.name)
        return default_dims

    @property
    def folds(self) -> int:
        return 5

    def generate_init_features(self, N: int, F: int) -> torch.Tensor:
        strategy = self.init_features['strategy']
        if strategy == 'dynamic_random':
            LOG.info('Generating init %s features N=%s, F=%s', strategy, N, F)
            return torch.rand(N, F)
        elif strategy == 'static_random':
            static_path = DATASET_DIR / f'random_features/{N}x{F}.pt'
            if static_path.exists():
                LOG.info('Loading static random features from %s', static_path)
                return torch.load(static_path)
            else:
                LOG.info('Generating static random features N=%s, F=%s', N, F)
                features = torch.rand(N, F)
                LOG.info('Saving static random features to %s', static_path)
                torch.save(features, static_path)
                return features
        elif strategy == 'vanilla':
            LOG.info('Generating init %s features N=%s, F=%s', strategy, N, F)
            return torch.ones(N, F)
        raise ValueError(f'Invalid strategy: {strategy}')

    def _load_data(self):
        return EgrDenseData.from_pickled(self.graph_path)
