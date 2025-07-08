import logging
import typing as ty
from argparse import Namespace
from datetime import datetime
from shutil import rmtree
from types import SimpleNamespace

import torch
from tqdm import tqdm

import egr.classifier as clf
import egr.explainer as exp
import egr.util as util
import workflows.config as wc

from egr.fsg import annotations as feature_generation


LOG = logging.getLogger(__name__)


def train(cfg: wc.WorkflowConfig, step_args, data, **kw):
    impl = step_args['implementation']
    LOG.debug('Classifier: %s', impl)
    fn: ty.Callable = globals()[f'train_{impl}']
    return fn(cfg, step_args, data, **kw)


def explain(cfg: wc.WorkflowConfig, step_args: ty.Dict, data, **kw) -> bool:
    impl = step_args['implementation']
    LOG.debug('Explainer: %s', impl)
    fn: ty.Callable = globals()[f'explain_{impl}']
    return fn(cfg, step_args, data, **kw)


def annotate(cfg: wc.WorkflowConfig, step_args: ty.Dict, data, **kw) -> bool:
    LOG.debug('step_args:%s', step_args)
    args = Namespace(**cfg.annotate_input(**kw), **step_args)
    timings = feature_generation.main(data, args)
    rmtree(args.tmp_dir)
    return True, timings


def explain_pyg(
    cfg: wc.WorkflowConfig, step_args: ty.Dict, data, **kw
) -> ty.Tuple[bool, ty.Dict]:
    train_params: ty.Dict = cfg.train_input(**kw)
    ckpt: ty.Dict = torch.load(train_params['ckpt_path'])
    saved_params = ckpt['cg']['params']
    train_params.update(saved_params)
    model = clf.make_model(data, train_params)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    begin = datetime.now()
    args = SimpleNamespace(**step_args, **kw)
    exp.explain(cfg, args, data, model, **kw)
    end = datetime.now()
    return True, {'explain': {'begin': begin, 'end': end}}


def hp_tune_enabled(args) -> bool:
    tuning_enabled = hasattr(args, 'hp_tuning') and args.hp_tuning['enabled']
    tunable_iter = (
        'rounds' not in args.reproducibility
        or args.iteration not in args.reproducibility['rounds']
    )
    return tuning_enabled and tunable_iter


def train_pyg(cfg: wc.WorkflowConfig, step_args, data, **kw):
    p = cfg.train_input(**kw)
    LOG.debug('step:%s, kw:%s', step_args, p)
    args = SimpleNamespace(**step_args, **p)

    default_params = args.hp_tuning.get('defaults', {})
    params = {**default_params}
    if hp_tune_enabled(args):
        c = clf.Classifier(data, args)
        best_params = args.prev_optuna_params
        if best_params == {}:
            LOG.info('HP-tuning enabled for iter:%d', args.iteration)
            best_params = c.tune_hyper_parameters(args)
            LOG.info('Using optimized params from %s', best_params)
        else:
            save_path = args.optuna_best_params_path
            LOG.info(
                'Using previous opt params %s to %s', best_params, save_path
            )
            util.save_json(best_params, save_path)
        params.update(best_params)
    else:
        util.save_json({'best_params': params}, args.optuna_params_path)
        LOG.info(
            'HP-tuning disabled for iter:%d, using default params: %s',
            args.iteration,
            params,
        )

    classifier = clf.Classifier(data, args)
    LOG.info('Training classifier with params: %s', params)
    classifier.train(params, save_ckpt=True, log_training=True)
    classifier.predict(data)
    return True, {'train': {'begin': datetime.now(), 'end': datetime.now()}}
