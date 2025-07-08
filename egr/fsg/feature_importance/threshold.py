import logging

import torch

LOG = logging.getLogger(__name__)


def median(data):
    return torch.median(data)


def mean(data):
    return torch.mean(data)


def maximum(data):
    return torch.max(data)
