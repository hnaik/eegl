# EEGL - An Iterative GNN Enhancement Framework
# Copyright (C) 2025 Harish Naik

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see
# <https://www.gnu.org/licenses/>.

import logging
import os
import shutil
import sys
import typing as ty
from datetime import datetime
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable

import egr
import egr.data.pyg_data_helpers as pyg_helpers
from egr import util
from workflows import tasks, config

LOG = logging.getLogger('run')


def start_iteration(cfg, sample_id, fold):
    try:
        for step in cfg.steps:
            LOG.info('Running step: %s', step['name'])
            func: Callable = getattr(tasks, step['type'])
            success = func(cfg, step, str(sample_id))
            if not success:
                return False
    except KeyboardInterrupt as err:
        LOG.warning('^C keyboard interrupt, %s', err)
        sys.exit(0)
    except (AttributeError, AssertionError, FileNotFoundError) as err:
        LOG.exception('%s application error', err)
        return False
    return True


def generate_run_configs(cfg) -> ty.List:
    cfgs: ty.List = []
    sample_id = 'default' if not hasattr(cfg, 'sample_id') else cfg.sample_id
    for fold in cfg.folds:
        cfgs.append(
            [
                {
                    'sample_id': sample_id,
                    'variant': cfg.dataset,
                    'fold': fold,
                    'iteration': iteration,
                }
                for iteration in cfg.iterations
            ]
        )
    return cfgs


PADDING = '=' * 32
ROUND_PADDING = '-' * 28


def round_marker(fold: int, rnd: int | str) -> str:
    return f'{ROUND_PADDING} FOLD:{fold}, ROUND:{rnd} {ROUND_PADDING}'


def main(args):
    egr.init_logging(level_name=args.log_level)
    gaston_bin = Path(
        os.getenv('GASTON_BIN_PATH', util.default_ext_bin('gaston'))
    )
    assert gaston_bin.exists(), f'{gaston_bin} not found'

    cfg = config.WorkflowConfig(args)
    id_file = cfg.output_root / '.experiment_id'
    LOG.info('Writing Experiment ID:%s, to file: %s', cfg.run_id, id_file)
    id_file.parent.absolute().mkdir(parents=True, exist_ok=True)
    id_file.write_text(cfg.run_id)

    cfg.variant_dir.mkdir(parents=True, exist_ok=True)

    logfile_name = f'{args.config.stem}-{egr.util.now_ts()}.log'
    logfile_path = cfg.variant_dir / logfile_name
    LOG.info('Writing to log file %s', logfile_path)
    file_handler = logging.FileHandler(logfile_path, mode='w+')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(egr.LOG_FMT))
    logging.getLogger().addHandler(file_handler)

    if hasattr(cfg, 'clean_root') and cfg.clean_root:
        if cfg.variant_dir.exists():
            LOG.info('Cleaning root directory %s', cfg.variant_dir)
            shutil.rmtree(cfg.variant_dir)

    last_iteration = cfg.iterations[-1]
    num_folds: int = (cfg.fold['end'] - cfg.fold.get('begin', 1)) + 1
    params_file = cfg.variant_dir / 'params.json'
    LOG.info('Saving params to %s', params_file)
    cfg.save(params_file)

    dataset = pyg_helpers.load_dataset(cfg.dataset)
    flows = generate_run_configs(cfg)
    interrupted = False
    for i, flow in enumerate(flows):
        if interrupted:
            break
        fold: int = cfg.folds[i % num_folds]
        LOG.debug('%s FOLD: %2d %s', PADDING, fold, PADDING)
        fold_begin = datetime.now()
        dataset_kw = cfg.dataset_kw(fold=fold, sample_id='default')
        interrupt = False
        for task_args in flow:
            if interrupt:
                break
            iteration = task_args['iteration']
            variant = task_args['variant']
            if isinstance(variant, dict):
                variant = list(variant.keys())[0]
            LOG.debug('task_args:%s, var:%s', task_args['variant'], variant)
            timings_dir = cfg.variant_dir / 'timings' / f'fold-{fold}'
            timings_dir.mkdir(parents=True, exist_ok=True)
            timings_path = timings_dir / f'{iteration}.json'
            timings = {}
            LOG.info(round_marker(fold, iteration))

            dataset_kw.update(
                {'annotated_feature_path': cfg.input_feature_path(**task_args)}
            )
            data = pyg_helpers.process_data(cfg, fold, dataset_kw, dataset[0])
            task_data: ty.Dict = {}
            for attr in cfg.steps:
                if interrupt:
                    break
                LOG.debug('attr:%s', attr)
                if iteration == last_iteration and attr['type'] != 'train':
                    LOG.info(
                        'Last iteration, skipping non-training step %s',
                        attr['type'],
                    )
                    break
                s = '|'.join([f'{k}:{v}' for k, v in task_args.items()])
                LOG.debug('FOLD:%2d,Step:%s:%s', fold, attr['name'], s)
                func: ty.Callable = getattr(tasks, attr['type'])
                try:
                    if 'classifier' in task_data:
                        attr.update({'classifier': task_data['classifier']})
                    ret, time_data = func(cfg, attr, data=data, **task_args)
                    for step_name, step_time in time_data.items():
                        timings.update({step_name: step_time})
                    if not ret:
                        break
                except FileNotFoundError as err:
                    LOG.error('%s', err)
                    raise
                except KeyboardInterrupt:
                    LOG.info('Keyboard interrupt, exiting')
                    interrupt = True
                    break
            LOG.info('Writing times to %s', timings_path)
            util.save_json(timings, timings_path)
            LOG.info(round_marker(fold, iteration))
        fold_end = datetime.now()
        LOG.info('FOLD:%2d duration: %s', fold, fold_end - fold_begin)
        LOG.info('%s FOLD: %2d %s', PADDING, fold, PADDING)
    LOG.info('Logs written to: %s', logfile_path)


def run_fold(cfg, sample_id, fold: int):
    for iteration in cfg.iterations:
        run_iteration(cfg, sample_id, iteration, fold)


def run_iteration(cfg, sample_id, iteration, fold):
    cfg.set_iteration(iteration)
    LOG.info('Starting sample:%s, iter:%s', sample_id, iteration)
    begin = datetime.now()
    success = start_iteration(cfg, sample_id, fold)
    end = datetime.now()
    if not success:
        LOG.error('exiting')
        sys.exit(1)
    LOG.info('Finished iteration %s in %s', iteration, end - begin)


if __name__ == '__main__':
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    parser = ArgumentParser()
    egr.add_log_argument(parser)
    parser.add_argument('-c', '--config', type=Path, required=True)
    parser.add_argument(
        '--run-defaults',
        type=Path,
        default='run_configs/run_defaults.yml',
    )
    parser.add_argument(
        '--pattern-master',
        type=Path,
        default='run_configs/pattern_master.yml',
    )
    cli_args = parser.parse_args()
    start = datetime.now()
    main(cli_args)
    LOG.info('Elapsed time %s', datetime.now() - start)
