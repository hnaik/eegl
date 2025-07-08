#!/bin/bash

set -euo pipefail

action=$1

if [ $action == "clear-output" ]; then
    for path in $(git --no-pager diff --name-only "*.ipynb"); do
        jupyter nbconvert --clear-output --inplace $path
    done
fi