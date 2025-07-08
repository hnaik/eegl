#!/bin/bash

set -euxo pipefail

export EEGL_SOLVER_PATH=/external/glasgow_subgraph_solver
/home/egr/egr/.venv/bin/python -m workflows.run -c $INPUT_FILE
