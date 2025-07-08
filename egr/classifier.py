import logging
import time
import typing as ty

import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data as pygdata
from optuna.trial import TrialState
from tqdm import tqdm

import egr.models as models
import egr.result as result
import egr.util as eu

LOG = logging.getLogger(__name__)
BAR_FORMAT = '{l_bar}{bar:50}{r_bar}{bar:-50b}'
optuna.logging.enable_propagation()
optuna.logging.disable_default_handler()
optuna.logging.set_verbosity(optuna.logging.WARNING)


class Classifier:
    def __init__(self, data, args):
        if (
            'rounds' in args.reproducibility
            and args.iteration in args.reproducibility['rounds']
        ):
            LOG.info('Fixing random seed to %d', args.reproducibility['seed'])
            torch.manual_seed(args.reproducibility['seed'])
        else:
            LOG.debug('Random seed not fixed')
            torch.seed()
        self.args = args
        self.device = get_device()
        self.data = data.to(self.device)
        self.model = None

        self.res = result.Result(
            train_index=self.data.train_index,
            val_index=self.data.val_index,
            test_index=self.data.test_index,
            eval_args={
                'average_strategy': self.args.average_strategy,
                'cm_normalize': self.args.confusion_matrix_normalize,
            },
        )

    def train(
        self,
        params: ty.Dict,
        save_ckpt=False,
        log_training=False,
    ):
        self.model = make_model(self.data, params)
        self.model.dropout = params['dropout']
        optimizer = make_optimizer(self.model, params['opt_params'])
        iterations = (
            tqdm(range(params['epochs']), bar_format=BAR_FORMAT)
            if log_training
            else range(params['epochs'])
        )
        ytrue = self.data.y.unsqueeze(0)

        train_mask = self.data.train_mask

        y = self.data.y
        if hasattr(self.data, 'y_noise'):
            y = self.data.y_noise

        perf_train = {}
        self.model.train()
        for _ in iterations:
            begin_time = time.time()
            optimizer.zero_grad()
            out = self.model(self.data.x, self.data.edge_index)
            loss = F.nll_loss(out[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()
            elapsed = time.time() - begin_time

            ypred = out.unsqueeze(0)

            if log_training:
                desc = self.res.compute_train(
                    ypred, ytrue, loss, elapsed, has_val=self.data.has_val_idx
                )
                iterations.set_description(desc)
            perf_train = self.res.compute_tr(ypred, ytrue, 'train')

        self.args.output_root.mkdir(parents=True, exist_ok=True)
        train_metrics_file = self.args.output_root / 'train_metrics.json'
        eu.save_metrics(self.res.train_df, train_metrics_file)

        self.model.eval()
        out_final = self.model(self.data.x, self.data.edge_index)
        if save_ckpt:
            cg_data = dict(
                pred=out_final.cpu().detach().unsqueeze(0).numpy(),
                feat=self.data.x.unsqueeze(0),
                label=self.data.y,
                train_idx=self.data.train_index,
                val_idx=self.data.val_index,
                test_idx=self.data.test_index,
                params=params,
            )
            save_checkpoint(self.model, optimizer, self.args, params, cg_data)

        if self.data.has_val_idx:
            perf_val = self.res.compute_tr(ypred, ytrue, 'val')
            return perf_val['f1_score']
        return perf_train['f1_score']

    def predict(self, data: pygdata.Data):
        ypred, ytrue = self.do_predict(data)
        perf_test = self.res.compute_test(ypred, ytrue)
        LOG.info(
            'TEST: fold=%d, round=%d, %s',
            self.args.fold,
            self.args.iteration,
            result.format_result(perf_test),
        )
        eu.save_json(perf_test, self.args.output_root / 'test_metrics.json')

    def do_predict(self, data):
        self.model.eval()
        model = self.model.to(get_device())
        data = data.to(get_device())
        out = model(data.x, data.edge_index)
        ypred = out.unsqueeze(0)
        ytrue = data.y.unsqueeze(0)
        return ypred, ytrue

    def tune_hyper_parameters(self, args) -> ty.Dict:
        kw = {'study_name': args.study_name, 'direction': 'maximize'}
        if args.hp_tuning.get('store_params', False):
            kw['storage'] = f'sqlite:///{args.run_id}-db.sqlite3'
            LOG.info('Storing hyper-parameters at: %s', kw['storage'])
        else:
            LOG.info('Not storing hyper-parameters')
        study = optuna.create_study(**kw)
        study.optimize(
            self.objective,
            n_trials=args.hp_tuning.get('n_trials', 10),
            callbacks=[early_stopping],
            show_progress_bar=True,
        )

        pruned_trials = study.get_trials(
            deepcopy=False, states=[TrialState.PRUNED]
        )
        complete_trials = study.get_trials(
            deepcopy=False, states=[TrialState.COMPLETE]
        )

        opt_results = {
            'finished_trials': len(study.trials),
            'pruned_trials': len(pruned_trials),
            'complete_trials': len(complete_trials),
            'best_trial': study.best_trial.value,
            'trials_df': study.trials_dataframe().to_dict(),
            'best_params': study.best_params,
        }

        LOG.info(
            'Optuna: value:%s, params: %s',
            study.best_trial.value * 100.0,
            format_params(study.best_params),
        )
        LOG.info('Saving optimization results to %s', args.optuna_params_path)
        eu.save_json(opt_results, args.optuna_params_path)

        best_params = {
            'dropout': study.best_params['dropout'],
            'epochs': study.best_params['epochs'],
            'hidden_dim': study.best_params['hidden_dim'],
            'opt_params': {
                'name': 'Adam',
                'params': {
                    'weight_decay': study.best_params['weight_decay'],
                    'lr': study.best_params['lr'],
                },
            },
        }
        LOG.info(
            'Saving best params: %s to %s',
            best_params,
            args.optuna_best_params_path,
        )
        eu.save_json(best_params, args.optuna_best_params_path)
        return best_params

    def objective(self, trial):
        dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-3, 1e-1)
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
        opt_params = {
            'name': 'Adam',  # 'SGD
            'params': {'weight_decay': weight_decay, 'lr': lr},
        }
        epoch_set = trial.suggest_int('epochs', 200, 1000)
        hidden_dim = trial.suggest_int('hidden_dim', 16, 48)

        params = {
            'opt_params': opt_params,
            'dropout': dropout,
            'epochs': epoch_set,
            'hidden_dim': hidden_dim,
        }
        return self.train(save_ckpt=False, params=params)

    def __repr__(self):
        return f'Classifier(model=str({self.model}), results={self.res})'


def early_stopping(study, trial):
    def fmt(v):
        if isinstance(v, float):
            return f'{v:.3f}'
        return v

    params = [f'{k}:{fmt(v)}' for k, v in trial.params.items()]
    params.extend(
        [
            f'best#:{fmt(study.best_trial.number)}',
            f'best-val:{fmt(study.best_trial.value)}',
        ]
    )

    params_str = '|'.join(params)
    LOG.debug(
        'Trial[%03d], obj:%.3f, params:%s',
        trial.number,
        trial.value,
        params_str,
    )
    if trial.value == 1.0:
        LOG.info(
            'Early stopping trial number:%d, objective:%.3f',
            trial.number,
            trial.value,
        )
        study.stop()


def make_model(data: pygdata.Data, params) -> nn.Module:
    return models.GCN(data, params).to(get_device())


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_optimizer(model, opt_params):
    return getattr(torch.optim, opt_params['name'])(
        model.parameters(), **opt_params['params']
    )


def save_checkpoint(model, optimizer, args, params, cg_dict: ty.Dict):
    args.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            epoch=params['epochs'],
            optimizer=optimizer,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            cg=cg_dict,
        ),
        args.ckpt_path,
    )


def format_results(loss, tr: ty.Dict, val: ty.Dict, dur) -> str:
    return 'L:%.4f|A:%.2f/%.2f|P:%.2f/%.2f|R:%.2f/%.2f|F:%.2f/%.2f|t:%.3f' % (
        loss.item(),
        tr['acc'],
        val['acc'],
        tr['prec'],
        val['prec'],
        tr['recall'],
        val['recall'],
        tr['f1_score'],
        val['f1_score'],
        dur,
    )


def format_params(params: ty.Dict) -> str:
    return 'dropout:%.4f|weight_decay:%.4f|lr:%.4f|epochs:%d' % (
        params['dropout'],
        params['weight_decay'],
        params['lr'],
        params['epochs'],
    )
