import logging
from types import SimpleNamespace

import optuna
import pandas as pd
from optuna import TrialState

from egr.util import save_metrics

LOG = logging.getLogger(__name__)


def objective(trial):
    dropout = trial.suggest_uniform('dropout', 0.0, 0.5)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-1)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    opt_args = SimpleNamespace(
        opt={
            'name': 'Adam',  # 'SGD
            'params': {'weight_decay': weight_decay, 'lr': lr},
        }
    )


def optimize():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    pruned_trials = study.get_trials(
        deepcopy=False, states=[TrialState.PRUNED]
    )
    complete_trials = study.get_trials(
        deepcopy=False, states=[TrialState.COMPLETE]
    )
    results = {
        'finished_trials': len(study.trials),
        'pruned_trials': len(pruned_trials),
        'complete_trials': len(complete_trials),
        'best_trial': study.best_trial.value,
    }
    df_results = pd.DataFrame(results, index=[0])
    LOG.info('Results: %s', df_results)
    df_params = pd.DataFrame(study.best_params, index=[0])
    LOG.info('Best params: %s', df_params)
    save_metrics(df_results, 'optuna_results.json')
    save_metrics(df_params, 'optuna_params.json')
