from __future__ import annotations

import logging
import os
import typing as ty
import yaml
from argparse import Namespace
from tempfile import mkdtemp
from pathlib import Path
from typing import Dict, List

import numpy as np

import egr
import egr.util as eu

LOG = logging.getLogger('config')


def make_variant_dir(cfg: ty.Dict) -> str:
    dir_s = cfg['data_class']
    if 'params' in cfg and 'name' in cfg['params']:
        return dir_s + '_' + cfg['params']['name']
    return dir_s


class RunConfig:
    def __init__(self, flow_cfg: WorkflowConfig, **kw):
        self.variant: ty.Optional[str] = ''
        self.__dict__.update(**flow_cfg.__dict__)
        self.__dict__.update(**kw)

        LOG.debug('RunConfig kw:%s', kw)

        self.iteration: int = kw.get('iteration', 0)
        self.idx_fname: str = f'{self.fold:02d}.json'
        self.input_tag: str = f'r{self.iteration}'
        self.iteration_dir: Path = self.variant_dir / self.input_tag
        self.dataset_dir: Path = self.input_root / make_variant_dir(
            self.dataset
        )
        self.index_path: Path = self.dataset_dir / 'indices' / self.idx_fname
        self.input_graph_path = self.dataset_dir / f'{self.sample_id}.json'
        self.input_label_path: Path = self.dataset_dir / 'labels.txt'
        self.feat_file_name: str = 'annotated_features.pt'
        self.input_dir: Path = self._io_dir(self.input_tag)
        self.output_path: Path = self._io_dir(self.output_tag)
        self.output_feature_path: Path = self.output_path / self.feat_file_name
        self.ckpt_path: Path = self.input_dir / f'{self.sample_id}.pt'
        self.train_dir: Path = self.input_dir / 'train'
        self.explain_dir = self.input_dir / 'explain'
        labels_file_name: str = f'predicted_labels-{self.sample_id}.txt'
        self.predicted_label_path: Path = self.input_dir / labels_file_name
        self.fsg_dir = self.input_dir / 'fsg'
        self.tmp_mined_sg = self.input_dir / 'tmp_mined_sg'
        self.optuna_params_path = self.train_dir / 'optuna_params.json'
        self.optuna_best_params_path = (
            self.train_dir / 'optuna_best_params.json'
        )

    @property
    def output_tag(self) -> str:
        if isinstance(self.iteration, int):
            return f'r{self.iteration + 1}'
        return f'r{self.iteration}_'

    @property
    def train_input(self) -> ty.Dict:
        study_name = (
            f'{self.dataset["data_class"].lower()}:'
            f'{self.dataset["params"]["name"].lower()}:f{self.fold}:'
            f'r{self.iteration}:{self.timestamp}'
        )
        return dict(
            index_file=self.index_path,
            input_graph_file=self.input_graph_path,
            input_label_file=self.input_label_path,
            annotated_feature_path=self.input_feature_path,
            ckpt_path=self.ckpt_path,
            output_root=self.train_dir,
            predicted_label_file=self.predicted_label_path,
            feature_importance=self.feature_importance,
            fold=self.fold,
            study_name=study_name,
            experiment_id=study_name,
            optuna_params_path=self.optuna_params_path,
            optuna_best_params_path=self.optuna_best_params_path,
            prev_optuna_params=self.prev_optuna_params,
            reproducibility=self.reproducibility,
            **self.global_attr,
        )

    def explain_input(self, node_id: int, **kw) -> ty.Dict:
        return dict(
            ckpt_file=self.ckpt_path,
            output_root=self.explain_dir,
            explain_node=node_id,
            graph_idx=-1,
            graph_mode=False,
            multigraph_class=-1,
            output_type='json',
            sample_id=self.sample_id,
            optuna_params=self.load_optuna_params(),
            **kw,
            **self.global_attr,
        )

    @property
    def annotate_input(self) -> Dict:
        params = dict(
            index_file=self.index_path,
            data_root=self.explain_dir,
            output_feature_file=self.output_feature_path,
            input_graph_file=self.input_graph_path,
            input_label_file=self.input_label_path,
            output_dir=self.output_path,
            fsg_dir=self.fsg_dir,
            prev_fsg_dir=self.prev_fsg_dir,
            tmp_mined_sg=self.tmp_mined_sg,
            freq=self.gaston_freq_threshold,
            predicted_label_file=self.predicted_label_path,
            data_dim=self.annotate_dim,
            tmp_dir=Path(mkdtemp()),
            timeout_secs=None,
            optuna_params=self.load_optuna_params(),
            **self.global_attr,
        )
        return params

    @property
    def prev_optuna_params(self) -> ty.Optional[Dict]:
        if self.iteration > 0:  # Only check for previous iterations
            input_dir = self._io_dir(f'r{self.iteration - 1}') / 'train'
            return eu.read_json(input_dir / 'optuna_best_params.json')
        return {}

    @property
    def input_feature_path(self) -> Path:
        if self.iteration == 0:
            return None
        elif self.iteration == 'L':
            return self.dataset_dir / 'label_features.npy'
        elif self.iteration == 'R':
            random_fname = f'random_{self.feat_file_name}'
            return self.input_root / 'features' / random_fname
        elif isinstance(self.iteration, str) and self.iteration[0] == 'P':
            return self.dataset_dir / self.iteration / f'{self.sample_id}.npy'
        return self.input_dir / self.feat_file_name

    @property
    def feature_importance(self) -> ty.Optional[np.ndarray]:
        if isinstance(self.iteration, int) and self.iteration > 0:
            itr = self.iteration - 1
            path = self._io_dir(f'r{itr}') / 'explain/feature_importance.pt'
            if path.exists():
                return eu.load_features(path)
            else:
                LOG.error('Feature importance file: %s not found', path)
        return None

    @property
    def global_attr(self) -> Dict[str, str]:
        nproc = self.nproc if hasattr(self, 'nproc') else os.cpu_count()
        return dict(
            run_id=self.run_id,
            iteration=self.iteration,
            input_tag=self.input_tag,
            gpu=self.gpu,
            nproc=nproc,
            hp_tuning=self.hp_tuning,
            annotate_dim=self.annotate_dim,
        )

    @property
    def prev_fsg_dir(self) -> List[Path]:
        if isinstance(self.iteration, int) and self.iteration > 0:
            return self._io_dir(f'r{self.iteration - 1}') / 'fsg'

    def _io_dir(self, tag: str) -> Path:
        return self.variant_dir / tag / self.sample_id / f'{self.fold:02d}'

    def load_optuna_params(self, path=None) -> ty.Dict:
        path = path or self.optuna_best_params_path
        return eu.read_json(path)


class WorkflowConfig:
    def __init__(self, args: Namespace):
        self.datestamp: str = eu.today_ts()
        self.timestamp: str = eu.now_ts()
        self.config_file_path: Path = args.config
        configs: Dict = yaml.safe_load(args.config.open())
        defaults: Dict = yaml.safe_load(args.run_defaults.open())

        d: ty.Dict = {**defaults}
        d.update(**configs)
        for idx, step in enumerate(configs['steps']):
            key = step['type']
            for k, v in defaults['steps'][key].items():
                if k not in step:
                    d['steps'][idx].update({k: v})
        self.__dict__.update(**d)
        if not hasattr(self, 'run_id') or self.run_id.lower() == 'auto':
            stem = self.config_file_path.stem
            self.run_id = f'{self.timestamp}_{stem}_{self.dataset_suffix}'
        elif hasattr(self, 'run_id'):
            if ':' in self.run_id:
                tag_type, tag_name = self.run_id.split(':')
                if tag_type == 'static_tag':
                    self.run_id = tag_name
                elif tag_type == 'date_tag':
                    self.run_id = self.datestamp + '_' + tag_name
                elif tag_type == 'datetime_tag':
                    self.run_id = self.timestamp + '_' + tag_name
                self.run_id += '_' + self.dataset_suffix

        self.output_root: Path = Path(self.output_root).expanduser().absolute()
        self.run_root: Path = self.output_root / self.run_id
        data_root: Path = Path(self.input_data_root).expanduser().absolute()
        self.input_root: Path = data_root / 'input_data'
        if not hasattr(self, 'timeout_secs'):
            self.timeout_secs = None
        self.variant_dir: Path = self.run_root / make_variant_dir(self.dataset)
        LOG.info('Input:%s, Output:%s', self.input_root, self.variant_dir)

    @property
    def dataset_suffix(self) -> str:
        suffix = self.dataset['data_class'].lower()
        if 'params' in self.dataset and 'name' in self.dataset['params']:
            suffix += '_' + self.dataset['params']['name'].lower()
        return suffix

    @property
    def folds(self) -> ty.List[int]:
        begin: int = self.fold.get('begin', 1)
        end: int = self.fold.get('end', self.fold.get('max', 10))
        return list(range(begin, end + 1))

    def set_iteration(self, iteration: int | str):
        self.iteration = iteration
        self.input_tag: str = f'r{self.iteration}'

    def _index_file_name(self, variant: str) -> str:
        if not hasattr(self, 'index_file_name'):
            m = self._master_details(variant)
            return f'indices-{m.total_size}.json'
        return self.index_file_name

    def _index_path(self, variant: str) -> Path:
        return self.root_path / self._index_file_name(variant)

    def train_input(self, **kw) -> Dict:
        kw.update({'timestamp': self.timestamp})
        cfg = RunConfig(self, **kw)
        return cfg.train_input

    def predicted_labels(self, variant: str, sample_id: str) -> Path:
        return (
            self.run_root
            / variant
            / self.input_tag
            / f'predicted_labels-{sample_id}.txt'
        )

    def explain_input(self, node_id: int, **kw: Dict):
        return self.explain_cfg(**kw).explain_input(node_id=node_id)

    def explain_cfg(self, **kw: ty.Dict) -> RunConfig:
        return RunConfig(self, **kw)

    def annotate_input(self, **kw: Dict) -> Dict:
        cfg = RunConfig(self, **kw)
        return cfg.annotate_input

    def output_feature_file(self, variant: str, sample_id: str) -> Path:
        return self.h_path(self.run_root, variant, sample_id)

    def h_path(self, root: Path, variant: str, sample_id: str) -> Path:
        h_dir = root / variant / self.output_tag / sample_id
        return h_dir / self._features_file_name(variant)

    def explain_subgraph_dir(self, variant: str, sample_id: str) -> Path:
        return self.explain_dir(variant, sample_id) / 'subgraph'

    def explain_feature_dir(self, variant: str, sample_id: str) -> Path:
        return self.explain_dir(variant, sample_id) / 'feature'

    def explain_dir(self, variant: str, sample_id: str) -> Path:
        return self.run_root / variant / self.input_tag / sample_id / 'explain'

    def fsg_dir(self, variant: str, sample_id: str) -> Path:
        return self.run_root / variant / self.input_tag / sample_id / 'fsg'

    def prev_fsg_dir(self, variant: str, sample_id: str) -> List[Path]:
        if isinstance(self.iteration, int) and self.iteration > 0:
            return (
                self.run_root
                / variant
                / f'r{self.iteration - 1}'
                / sample_id
                / 'fsg'
            )

    def histogram_dir(self, variant: str, sample_id: str) -> Path:
        return self.run_root / variant / self.input_tag / 'hist'

    def input_path(self, variant: str, sample_id: str) -> Path:
        return self.input_dir(variant) / f'{self.stem(sample_id)}.json'

    def input_graph_path(self, variant: str, sample_id: str) -> Path:
        return self.input_root / variant / f'{sample_id}.json'

    def input_label_path(self, variant: str) -> Path:
        return self.input_root / variant / 'labels.txt'

    def input_dir(self, variant: str) -> Path:
        return self.input_root / variant / 'input'

    def ckpt_path(self, variant: str, sample_id: str) -> Path:
        return self.run_root / variant / self.input_tag / f'{sample_id}.pt'

    def train_dir(self, variant: str, sample_id: str) -> Path:
        return self.run_root / variant / self.input_tag / sample_id / 'train'

    def stem(self, sample_id: str) -> str:
        return f'{self.data_prefix}-{sample_id}'

    @property
    def experiment_id(self) -> str:
        return f'{self.run_id}:{self.dataset["data_class"].lower()}'

    def __str__(self) -> Dict:
        return egr.util.to_string(self.__dict__)

    def dataset_kw(self, **kw) -> Dict:
        cfg = RunConfig(self, **kw)
        return cfg.train_input

    def save(self, path: Path):
        with path.open('w') as f:
            f.write(egr.util.to_json(self.__dict__))

    def input_feature_path(self, **kw) -> Path:
        cfg = RunConfig(self, **kw)
        return cfg.input_feature_path
