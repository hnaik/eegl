import logging
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

from tqdm import tqdm

import egr
import egr.data.pyg_data_helpers as pdh
import egr.classifier as clf
import egr.result as result

LOG = logging.getLogger('logic_property')


def main(args):
    LOG.info('Running main with args: %s', args)
    dataset_cfg = {
        'src': 'egr',
        'data_class': 'Grid2DLogicDataset',
        'num_samples': 50,
        'data_params': {'rows': 20, 'cols': 20, 'frac': 0.15},
        'params': {
            'name': args.name,
            'root': 'data',
            'transform': 'egr_logic',
            'pre_transform': None,
            'pre_filter': None,
        },
    }

    now_ts = egr.util.now_ts()
    run_id = now_ts if args.run_id is None else now_ts + '_' + args.run_id

    output_dir = (
        args.output_root
        / dataset_cfg['data_class']
        / dataset_cfg['params']['name']
        / run_id
    )
    spec = SimpleNamespace(
        fold=1,
        iteration=0,
        average_strategy='weighted',
        confusion_matrix_normalize='all',
        output_root=output_dir,
        reproducibility={'seed': 42},
    )
    train_params = {
        'epochs': 1000,
        'hidden_dim': 16,
        'dropout': 0.5,
        'opt_params': {
            'name': 'Adam',
            'params': {'lr': 0.01, 'weight_decay': 5e-4},
        },
    }
    dataset = pdh.load_dataset(dataset_cfg)

    if args.order == 'asc-node':
        dataset = sorted(dataset, key=lambda x: x.G.number_of_nodes())
    elif args.order == 'asc-edge':
        dataset = sorted(dataset, key=lambda x: x.G.number_of_edges())
    elif args.order == 'desc-node':
        dataset = sorted(
            dataset, key=lambda x: x.G.number_of_nodes(), reverse=True
        )
    elif args.order == 'desc-edge':
        dataset = sorted(
            dataset, key=lambda x: x.G.number_of_edges(), reverse=True
        )

    for fold in range(1, 6):
        run_experiment_for_fold(dataset_cfg, spec, train_params, dataset, fold)


def run_experiment_for_fold(dataset_cfg, spec, train_params, dataset, fold):
    data = pdh.process_data(dataset_cfg, fold, {}, dataset[0])

    LOG.info('[Fold:%03d] Training with Graph:%s', fold, data.G)
    classifier = clf.Classifier(data, spec)
    classifier.train(train_params, save_ckpt=False, log_training=True)

    results_set = []
    pbar = tqdm(dataset)
    for i, data_point in enumerate(pbar):
        pbar.set_description(f'Predicting for Graph {i}')
        data_point = pdh.process_data(dataset_cfg, fold, {}, data_point)
        res = result.Result(
            train_index=data_point.train_index,
            val_index=data_point.val_index,
            test_index=data_point.test_index,
            eval_args={
                'average_strategy': spec.average_strategy,
                'cm_normalize': spec.confusion_matrix_normalize,
            },
        )
        ypred, ytrue = classifier.do_predict(data_point)
        perf_test = res.compute_test(ypred, ytrue)
        LOG.debug('Performance: %s', result.format_result(perf_test))
        results_set.append(
            {
                'ID': i + 1,
                '#Nodes': data_point.G.number_of_nodes(),
                '#Edges': data_point.G.number_of_edges(),
                **perf_test,
            }
        )
    output_path = spec.logdir / f'results_fold-{fold:02}.json'
    LOG.info('Saving results to %s', output_path)
    egr.util.save_json(results_set, output_path)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--name', type=str, required=True)
    parser.add_argument(
        '--log-level',
        type=str,
        default='info',
        choices=['debug', 'info', 'warning', 'error', 'critical'],
    )
    parser.add_argument('--output-root', type=Path, default='.output')
    parser.add_argument(
        '--order',
        type=str,
        default='unordered',
        choices=[
            'asc-node',
            'asc-edge',
            'desc-node',
            'desc-edge',
            'unordered',
        ],
    )
    parser.add_argument('--run-id', type=str)
    parsed_args = parser.parse_args()
    egr.init_logging(level_name=parsed_args.log_level)
    main(parsed_args)
