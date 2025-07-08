import logging

import torch
import numpy as np
from sklearn import metrics

LOG = logging.getLogger(__name__)


def eval_set(ytrue, ypred, average_strategy, cm_normalize):
    ytrue: np.array = ytrue.cpu().numpy()
    ypred: np.array = ypred.cpu().numpy()
    prec, recall, f_score, support = metrics.precision_recall_fscore_support(
        ytrue, ypred, average=average_strategy, zero_division=0
    )
    extra = {}
    if cm_normalize is not None:
        cm = metrics.confusion_matrix(ytrue, ypred, normalize=cm_normalize)
        extra.update({f'conf_mat_normalized_{cm_normalize}': cm})
    labels = np.sort(np.unique(ytrue))
    return {
        'acc': metrics.accuracy_score(ytrue, ypred),
        'balanced_acc': metrics.balanced_accuracy_score(ytrue, ypred),
        'prec': prec,
        'recall': recall,
        'f1_score': f_score,
        'support': support,
        'conf_mat': metrics.confusion_matrix(ytrue, ypred, labels=labels),
        'multi_cm': metrics.multilabel_confusion_matrix(ytrue, ypred),
        **extra,
    }


@torch.no_grad()
def eval_test(ypred, ytrue, indices, average_strategy, cm_normalize):
    pred_labels = torch.argmax(ypred, 2)
    y_pred = torch.ravel(pred_labels[:, indices])
    y_true = torch.ravel(ytrue[:, indices])
    try:
        y_pred = torch.ravel(pred_labels)
        y_true = torch.ravel(ytrue)
    except IndexError as err:
        LOG.error(
            'ypred:%s, ytrue:%s, indices:%s, %s',
            ypred.shape,
            ytrue.shape,
            indices,
            err,
        )
    # y_pred = torch.ravel(pred_labels[:, indices])
    # y_true = torch.ravel(ytrue[:, indices])

    return eval_set(y_true, y_pred, average_strategy, cm_normalize)


def eval_train(ypred, ytrue, indices, average_strategy, cm_normalize):
    pred_labels = torch.argmax(ypred, 2)
    y_pred = torch.ravel(pred_labels[:, indices])
    y_true = torch.ravel(ytrue[:, indices])

    return eval_set(y_true, y_pred, average_strategy, cm_normalize)


def evaluate_train(ypred, ytrue, indices, average_strategy, cm_normalize):
    pred_labels = torch.argmax(ypred, 2)
    y_pred = torch.ravel(pred_labels[:, indices])
    y_true = torch.ravel(ytrue[:, indices])

    return eval_set(y_true, y_pred, average_strategy, cm_normalize)
