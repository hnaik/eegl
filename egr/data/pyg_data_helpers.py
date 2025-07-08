import logging
import typing as ty
import urllib.error

import networkx as nx
import torch
import torch_geometric.utils as pygutils
from torch_geometric import data as pygdata
from torch_geometric import datasets

import egr.datasets
import egr.util as util
from egr.data import transforms

LOG = logging.getLogger(__name__)


def load_dataset(cfg: ty.Dict) -> pygdata.Data:
    params = normalize_params(cfg)
    dataset = get_data_class(cfg)(cfg, **params)
    return dataset


def process_data(cfg, fold, kw, data):
    LOG.debug('fold:%d, cfg=%s', fold, cfg)
    LOG.debug('kw=%s', kw)

    def get_mask(name) -> ty.Dict[str, torch.BoolTensor]:
        m_ = getattr(data, f'{name}_mask')
        if len(m_.shape) == 1:
            m_ = m_.unsqueeze(1)
        mask = m_[:, fold - 1]
        return {
            f'{name}_mask': mask,
            f'{name}_index': mask.nonzero().squeeze(1),
        }

    try:
        params = {
            **get_mask('train'),
            **get_mask('val'),
            **get_mask('test'),
        }
        if hasattr(data, 'y_noise'):
            params.update({'y_noise': data.y_noise[:, fold - 1]})
        if hasattr(data, 'x_orig'):
            params['x_orig'] = data.x_orig

        params['x'] = process_features(data, kw)

        return pygdata.Data(
            G=make_graph(data),
            y=data.y,
            edge_index=data.edge_index,
            num_classes=torch.unique(data.y).shape[0],
            num_node_features=data.num_node_features,
            total_size=data.y.shape[0],
            has_val_idx=getattr(data, 'has_val_idx', False),
            **params,
        )
    except (TypeError, AttributeError, urllib.error.HTTPError) as e:
        LOG.error('%s cfg:%s, data:%s', e, cfg, data)
        raise


def process_features(data, kw) -> torch.Tensor:
    h = load_annotated_features(kw)
    if hasattr(data, 'x_orig') and h is not None:
        x = torch.cat((data.x_orig, h), dim=1)
        LOG.debug(
            'adding %s annotated to original %s, res %s',
            h.shape,
            data.x_orig.shape,
            x.shape,
        )
        return x
    LOG.debug(
        'No annotation(h=%s) or input features(x_orig=%s)',
        h is not None,
        hasattr(data, 'x_orig'),
    )
    return data.x


def make_graph(data) -> nx.Graph:
    G = pygutils.to_networkx(data, to_undirected=True)
    G.graph['y'] = data.y
    G.graph['x'] = data.x
    for i, node in enumerate(G.nodes):
        G.nodes[node]['x'] = data.x[i].tolist()
        G.nodes[node]['y'] = data.y[i].tolist()
    return G


def normalize_params(cfg: ty.Dict) -> ty.Dict:
    kw = {**cfg.get('params', {})}
    if 'root' not in kw:
        root = util.DATASET_DIR / cfg['src'] / cfg['data_class']
        if 'name' in kw:
            root /= kw['name']
        kw['root'] = root

    if 'log' not in kw:
        kw['log'] = False

    if 'transform' in kw and isinstance(kw['transform'], str):
        kw['transform'] = getattr(transforms, kw['transform'])

    return kw


def load_annotated_features(cfg) -> ty.Optional[util.FeatureType]:
    try:
        if 'annotated_feature_path' in cfg:
            path = cfg['annotated_feature_path']
            if path and path.exists():
                LOG.info('Loading annotated features from %s', path)
                return util.load_features(path)
            return None
        LOG.debug('No annotated feature path defined')
    except FileNotFoundError:
        LOG.info('No feature file found, continuing')
    return None


def get_data_class(cfg: ty.Dict) -> ty.Type:
    src = cfg.get('src', 'pyg')
    match src:
        case 'pyg':
            return getattr(datasets, cfg['data_class'])
        case 'egr':
            data_class = getattr(egr.datasets, cfg['data_class'])
            data_class.init_features = cfg.get(
                'init_features', {'dim': 1, 'strategy': 'vanilla'}
            )
            return data_class
    raise RuntimeError(f'Unknown data source {src}')
