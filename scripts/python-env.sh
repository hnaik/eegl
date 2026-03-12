#!/usr/bin/env bash

set -euo pipefail

root_dir=$(realpath $(dirname $0)/..)
VENV=${root_dir}/.venv
option=$1

if [ $option == "create" ]; then
    if [[ ! -d ${VENV} ]]; then
        uv sync
    fi
elif [ $option == "update" ]; then
    uv sync
else
    echo "Invalid option"
    exit 1
fi
