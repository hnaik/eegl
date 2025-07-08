import json
from io import StringIO
from unittest.mock import Mock

import numpy as np
import torch

from egr import util


def test_load_labels():
    label_input = StringIO('0,0,0,0,1,2,3,4,1,3,2,4')
    expected = torch.LongTensor([0, 0, 0, 0, 1, 2, 3, 4, 1, 3, 2, 4])
    actual = util.load_labels(label_input)
    torch.testing.assert_close(actual, expected)
