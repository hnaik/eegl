#!/bin/bash

set -euo pipefail

root_dir=$(realpath $(dirname $0)/..)
py=$root_dir/.venv/bin/python

$py -c "import torch; print('CUDA available %s ' % torch.cuda.is_available())"
$py -m torch.utils.collect_env
