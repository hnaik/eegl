import logging

from egr.datasets.egr_base_dataset import EgrBaseDataset

LOG = logging.getLogger(__name__)


class Orbits(EgrBaseDataset):
    @property
    def dim(self) -> int:
        if self.init_features['dim'] > 0:
            LOG.info('Using spec feature_dim=%s', self.init_features['dim'])
            return self.init_features['dim']
        return 10

    @property
    def folds(self) -> int:
        return 10

    def _processed_file_names(self):
        return [f'processed-dim{self.dim}.pt']
