import logging
import typing as ty
from argparse import ArgumentParser
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import yaml
from tqdm import tqdm

import egr
import egr.data.pyg_data_helpers as pdh
import egr.classifier as clf
import egr.result as result
import egr.util as eu
import egr.graph_utils as gu

LOG = logging.getLogger('logic.mixed')


def make_dataset(cfg_list, params) -> ty.List:
    all_datasets = []
    for cfg in cfg_list:
        cfg.update({'params': params})
        dataset = pdh.load_dataset(cfg)
        all_datasets.append(dataset)
    LOG.info('Loaded %d datasets', len(all_datasets))
    return all_datasets


def run_id(cfg: SimpleNamespace) -> str:
    if not hasattr(cfg, 'run_id'):
        path = cfg.config_file_path
        LOG.info('config_file_path: %s', cfg.config_file_path)
        parent = str(path.parent).replace('/', '-')
        run_id = f'{parent}-{path.stem}'
        LOG.info('run_id: %s', run_id)
    LOG.info('run_id: %s', cfg['run_id'])


class ExperimentId:
    def __init__(self, cfg: ty.Dict):
        self.cfg = SimpleNamespace(**cfg)
        self.experiment_id = self.make_experiment_id()

    def make_experiment_id(self) -> str:
        return f'{self._get_prefix()}_{self._get_suffix()}'

    def _get_prefix(self) -> str:
        return datetime.now().strftime('%Y%m%d_%H%M%S')

    def _get_suffix(self) -> str:
        if not hasattr(self.cfg, 'run_id'):
            path = self.cfg.config_file_path
            parent = str(path.parent).replace('/', '-')
            current_run_id = f'{parent}-{path.stem}'
            LOG.info(
                'run_id not defined, generated experiment Id: %s',
                current_run_id,
            )
            return current_run_id
        return self.cfg.run_id


class Experiment:
    def __init__(self, cfg):
        self.cfg = cfg
        self.spec = SimpleNamespace(**cfg['spec'])

        self.output_dir = cfg['spec']['output_root'] / cfg['run_id']
        self.train_params = cfg['train_params']
        self.train_dataset_id = self.train_params['dataset_id']
        self.train_data_id = self.train_params['data_id']
        self.rule_config = cfg['rule_config']

    def make_output_dir(self, cfg: ty.Dict, fold: int) -> Path:
        return self.output_dir / cfg['data_class'] / f'fold-{fold:02}'

    def run_experiment(self, structures: ty.List, fold: int) -> ty.Dict:
        train_cfg = structures[self.train_dataset_id]
        output_dir = self.make_output_dir(train_cfg, fold)
        LOG.info(
            'Training using: %s, output dir:%s',
            train_cfg['data_class'],
            output_dir,
        )
        all_datasets = make_dataset(structures, self.rule_config)
        train_data = self.make_training_data(fold, train_cfg, all_datasets)
        args = deepcopy(self.spec)
        args.output_root = output_dir
        classifier = self.train(train_data, args)
        output_dir.mkdir(parents=True, exist_ok=True)

        inference_data = []
        begin = datetime.now()
        for data_idx, dataset in enumerate(all_datasets):
            data_cfg = structures[data_idx]
            results_set = []
            pbar = tqdm(dataset)
            for i, data_point in enumerate(pbar):
                pbar.set_description(
                    f'Running inference on {data_cfg["data_class"]} fold:{fold}, '
                    f'dataset {data_idx}: {i + 1}/{len(dataset)}, '
                )
                data_point = pdh.process_data(data_cfg, fold, {}, data_point)
                res = result.Result(
                    train_index=data_point.train_index,
                    val_index=data_point.val_index,
                    test_index=data_point.test_index,
                    eval_args={
                        'average_strategy': self.spec.average_strategy,
                        'cm_normalize': self.spec.confusion_matrix_normalize,
                    },
                )
                ypred, ytrue = classifier.do_predict(data_point)
                perf_test = res.compute_test(ypred, ytrue)
                results_set.append(
                    {
                        'ID': i + 1,
                        '#Nodes': data_point.G.number_of_nodes(),
                        '#Edges': data_point.G.number_of_edges(),
                        **perf_test,
                    }
                )
            output_path = output_dir / f'results_{data_idx:02}.json'
            data = {
                'params': structures[data_idx],
                'results': results_set,
            }
            LOG.info('Saving results to %s', output_path)
            egr.util.save_json(data, output_path)
            inference_data.append(data)
        train_g = self.save_training_graph(output_dir, train_data)
        LOG.info('Finished fold %d in %s', fold, datetime.now() - begin)
        return {
            'train': {
                'graph': train_g,
                'results_df': classifier.res.train_df,
            },
            'inference': inference_data,
        }

    def save_training_graph(self, output_dir: Path, data) -> nx.Graph:
        path = output_dir / 'training_graph.json'
        LOG.info('Saving training graph to %s', path)
        G = pdh.make_graph(data)
        graphlets = {k: gu.num_unique_motifs(G, k) for k in range(3, 7)}
        G.graph['graphlets'] = graphlets

        egr.util.save(G, path)
        return G

    def make_training_data(self, fold, cfg, all_datasets: ty.List):
        data = all_datasets[self.train_dataset_id][self.train_data_id]
        return pdh.process_data(cfg, fold, {}, data)

    def train(self, data, args) -> clf.Classifier:
        begin = datetime.now()
        classifier = clf.Classifier(data, args)
        classifier.train(self.train_params, save_ckpt=False, log_training=True)
        LOG.info('Training took %s', datetime.now() - begin)
        return classifier


def main(args):
    LOG.debug('Running main with args: %s', args)
    begin = datetime.now()
    cfg_data = load_config(args.config)
    cfg = SimpleNamespace(**cfg_data)
    structures = cfg.structures
    exp_id = ExperimentId(cfg_data)
    LOG.info('Starting experiment %s', exp_id.experiment_id)
    exp_data = OrderedDict()
    for _ in range(len(cfg.structures)):
        rot_cfg = {'run_id': exp_id.experiment_id, **cfg_data}
        rot_cfg['structures'] = structures
        exp = Experiment(rot_cfg)
        train_cfg = structures[exp.train_dataset_id]
        config_path = exp.output_dir / train_cfg['data_class'] / 'config.json'
        LOG.info('Saving config to : %s', config_path)
        egr.util.save_json(rot_cfg, config_path)
        folds_data = OrderedDict()
        for fold in range(cfg.folds['begin'], cfg.folds['end'] + 1):
            folds_data.update({fold: exp.run_experiment(structures, fold)})

        exp_data[train_cfg['data_class']] = {
            'config': train_cfg,
            'folds_data': folds_data,
            'structure_config': rot_cfg,
        }
        structures = rotate(structures)
    LOG.info('Finished in %s', datetime.now() - begin)

    pkl_path = exp.output_dir / 'results.pkl'
    LOG.info('Saving results to %s', pkl_path)
    eu.save_pickle(exp_data, pkl_path)


def rotate(items) -> ty.List:
    return items[1:] + items[:1]


def load_config(cfg_path: Path) -> ty.Dict:
    data = yaml.safe_load(cfg_path.read_text())
    LOG.info('Loaded config from %s', data['spec'])
    data['spec']['output_root'] = Path(data['spec']['output_root']).absolute()
    data['config_file_path'] = cfg_path
    return data


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=Path, required=True)
    parser.add_argument(
        '-l',
        '--log-level',
        type=str,
        default='info',
        choices=['debug', 'info', 'warning', 'error'],
    )
    parsed_args = parser.parse_args()
    egr.init_logging(level_name=parsed_args.log_level)
    main(parsed_args)
