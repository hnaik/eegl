from __future__ import annotations

import pickle
import platform
import json
import logging
import requests
import typing as ty
from argparse import ArgumentParser
from ctypes import Union
from datetime import date, datetime, timedelta
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import networkx as nx
import numpy as np
import pandas as pd
import torch
import yaml
from torch import Tensor, LongTensor

from egr.log import init_logging


LOG = logging.getLogger(__name__)

PROJECT_ROOT_DIR: Path = Path(__file__).parent.parent.absolute()
EXT_BIN_DIR: Path = (
    PROJECT_ROOT_DIR / 'external' / platform.system() / platform.processor()
)
DEPS_LIB_DIR: Path = PROJECT_ROOT_DIR / '.deps/lib'
DATASET_DIR: Path = PROJECT_ROOT_DIR / 'dataset'


class Paths:
    DATASET: Path = PROJECT_ROOT_DIR / 'dataset'
    RUN_CONFIGS_DIR: Path = PROJECT_ROOT_DIR / 'run_configs'


def read_json(path: Path) -> Dict:
    """Read JSON file from path"""
    LOG.debug('Loading path %s', path)
    return json.load(path.open())


def load_graph(path: Path) -> nx.Graph:
    """Load nx.Graph from json file

    Parameters
    ----------
    path : pathlib.Path
        Path to the JSON file

    Returns
    -------
    nx.Graph
        The graph object loaded from file

    """
    data = read_json(path)
    try:
        return nx.json_graph.node_link_graph(data, edges='edges')
    except KeyError as err:
        LOG.warning('KeyError: %s', data.keys())
    return nx.json_graph.node_link_graph(data, edges='links')


def now_ts(fmt: str = '%Y%m%d-%H%M%S') -> str:
    """Make now string"""
    return datetime.now().strftime(fmt)


def today_ts(fmt: str = '%Y%m%d') -> str:
    """Make today string"""
    return date.today().strftime(fmt)


class IoEncoder(json.JSONEncoder):
    def default(self, o: ty.Any) -> ty.Any:
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.float32):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, Tensor):
            return o.cpu().detach().tolist()
        elif isinstance(o, Path):
            return str(o)
        elif isinstance(o, datetime):
            return o.isoformat()
        elif isinstance(o, timedelta):
            return o.total_seconds()
        elif isinstance(o, type(o)):
            return o.__name__
        return super().default(o)


def to_dict(G: nx.Graph) -> Dict:
    if isinstance(G, nx.Graph):
        return nx.json_graph.node_link_data(G)
    elif isinstance(G, dict):
        return G
    raise RuntimeError(f'Unsupported type {type(G)}')


def to_json(G: nx.Graph, **kwargs) -> str:
    """Convert graph to JSON

    Parameters
    ----------
    G : networkx.Graph
        Graph object
    **kwargs : dict
        Keyword arguments

    Returns
    -------
    str
        JSON string

    """
    return to_string(to_dict(G), **kwargs)


def to_string(data: Dict, **kwargs) -> str:
    return json.dumps(data, cls=IoEncoder, **kwargs)


def save(G: nx.Graph, path: Path, **kwargs):
    """Save graph to given path

    Parameters
    ----------
    G : nx.Graph
        Graph object
    path : pathlib.Path
        Path to save the graph to
    **kwargs: dict
        json.dump keyword args

    """
    data = to_dict(G)
    save_json(data, path)


def make_args(**kw) -> SimpleNamespace:
    return SimpleNamespace(**kw)


def normalize_path(path: Union[str, Path]) -> Path:
    path = (
        (path if isinstance(path, Path) else Path(path))
        .expanduser()
        .absolute()
    )
    return path


def save_json(data: List | Dict, path: Path, **kwargs):
    try:
        path = normalize_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w') as f:
            json.dump(data, f, cls=IoEncoder, **kwargs)
            return path
    except TypeError as e:
        LOG.error('%s\n%s', e, data)


def load_indices(path: Path) -> Dict[str, np.ndarray]:
    """Load indices as numpy arrays"""
    indices = read_json(path)
    return {k: np.array(v) for k, v in indices.items()}


def load_graph_features(data: Any) -> np.array:
    G: nx.Graph = None
    if isinstance(data, Path) or isinstance(data, str):
        return load_graph_features(load_graph(data))
    elif isinstance(data, nx.Graph):
        G = data
    else:
        raise RuntimeError(f'{type(data)} is not supported')
    return np.array([G.nodes[n]['feat'] for n in G.nodes()])


def app_config(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        parser = ArgumentParser()
        parser.add_argument(
            '--log-level',
            type=str,
            default='debug',
            choices=['debug', 'info', 'warning', 'error', 'critical'],
        )
        f(parser)
        cfg = parser.parse_args()
        init_logging(level_name=cfg.log_level)
        return cfg

    return wrapper


FeatureType: ty.TypeAlias = np.ndarray | Tensor


def save_features(fpath: Path, data: FeatureType):
    if fpath.suffix == '.npy':
        np.save(fpath, data)
    elif fpath.suffix == '.txt':
        np.savetxt(fpath, data)
    elif fpath.suffix == '.csv':
        np.savetxt(fpath, data, delimiter=',')
    elif fpath.suffix == '.pt':
        if isinstance(data, np.ndarray):
            LOG.debug('Saving tensor to %s after conversion', fpath)
            torch.save(torch.from_numpy(data), fpath)
        torch.save(data, fpath)
    else:
        raise RuntimeError(f'Unsupported file type {fpath.suffix}')


def load_features(fpath: Path) -> FeatureType:
    if fpath.suffix == '.npy':
        return np.load(fpath, allow_pickle=True)
    elif fpath.suffix == '.txt':
        return np.loadtxt(fpath)
    elif fpath.suffix == '.csv':
        return np.loadtxt(fpath, delimiter=',')
    elif fpath.suffix == '.pt':
        t = torch.load(fpath)
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t)
        return t.to(dtype=torch.float32)
    else:
        raise RuntimeError(f'Unsupported file type {fpath.suffix}')


def load_labels(path: Path) -> Tensor:
    return Tensor([int(l) for l in path.read().split(',')]).type(LongTensor)


def save_labels(data: List, path: Path):
    path.open('w').write(','.join([str(l) for l in data]))


def default_ext_bin(name: str) -> Path:
    return EXT_BIN_DIR / name


def load_variant_specs() -> ty.Dict:
    spec_file = PROJECT_ROOT_DIR / 'run_configs/pattern_master.yml'
    return yaml.safe_load(spec_file.open())


def load_variant_spec(name: str) -> ty.Dict | SimpleNamespace:
    args = SimpleNamespace(**load_variant_specs()[name])
    args.details = SimpleNamespace(**args.details)
    return args


def save_metrics(df: pd.DataFrame, path: Path):
    LOG.debug('Saving %s', path)
    df.to_json(path)


def download(url: str, dest: Path, overwrite: bool = False):
    if dest.exists() and not overwrite:
        LOG.info('File exists, skipping download')
        return dest
    LOG.info('Downloading %s to %s', url, dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, allow_redirects=True)
    dest.write_bytes(r.content)
    return dest


def save_pickle(data: Any, path: Path):
    pickle.dump(data, path.open('wb'))


def load_pickle(path: Path) -> ty.Dict:
    return pickle.load(path.open('rb'))
