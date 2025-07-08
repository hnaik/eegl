import logging
import typing as ty

import pandas as pd

import egr.evaluation as evl

LOG = logging.getLogger(__name__)


class Result:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.eval_args = getattr(self, 'eval_args', {})
        self._perf = {'train': [], 'val': [], 'test': []}

    @property
    def train_df(self):
        return pd.DataFrame(self._perf['train'])

    @property
    def val_df(self):
        return pd.DataFrame(self._perf['val'])

    def compute_train(self, ypred, ytrue, loss, dur, has_val=True):
        tr = self.compute_tr(ypred, ytrue, 'train')
        if not has_val:
            return format_train_result(loss, tr, dur)
        val = self.compute_tr(ypred, ytrue, 'val')
        return format_train_results(loss, tr, val, dur)

    def compute_tr(self, ypred, ytrue, name):
        indices = getattr(self, f'{name}_index')
        result = evl.eval_train(ypred, ytrue, indices, **self.eval_args)
        self._perf[name].append(result)
        return result

    def compute_test(
        self, ypred, ytrue, indices: ty.Optional[ty.List[int]] = None
    ):
        indices = indices or getattr(self, 'test_index')
        result = evl.eval_test(ypred, ytrue, indices, **self.eval_args)
        self._perf['test'].append(result)
        return result

    def __repr__(self):
        return str(self.__dict__)


def format_train_result(loss, tr: ty.Dict, dur) -> str:
    return 'L:%.4f|A:%.2f|P:%.2f|R:%.2f|F:%.2f|t:%.3f' % (
        loss.item(),
        tr['acc'] * 100,
        tr['prec'] * 100,
        tr['recall'] * 100,
        tr['f1_score'] * 100,
        dur,
    )


def format_train_results(loss, tr: ty.Dict, val: ty.Dict, dur) -> str:
    return 'L:%.4f|A:%.2f/%.2f|P:%.2f/%.2f|R:%.2f/%.2f|F:%.2f/%.2f|t:%.3f' % (
        loss.item(),
        tr['acc'] * 100,
        val['acc'] * 100,
        tr['prec'] * 100,
        val['prec'] * 100,
        tr['recall'] * 100,
        val['recall'] * 100,
        tr['f1_score'] * 100,
        val['f1_score'] * 100,
        dur,
    )


def format_result(res) -> str:
    return 'Acc:%.3f|Prec:%.3f|Recall:%.3f|F1-Score:%.3f' % (
        res['acc'] * 100,
        res['prec'] * 100,
        res['recall'] * 100,
        res['f1_score'] * 100,
    )
