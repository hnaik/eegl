#!/usr/bin/env bash

set -euo pipefail

root_dir=$(realpath $(dirname $0)/..)
VENV=${root_dir}/.venv
option=$1
env_file=environment.yml

if [[ $# -eq 2 ]] && [[ $2 == "cpu" ]]; then
    env_file=environment-cpu.yml
fi


if [ $option == "create" ]; then
    if [[ ! -d ${VENV} ]]; then
        mamba env create -p ${VENV} -f $env_file
    fi
elif [ $option == "update" ]; then
    mamba env update -p ${VENV} -f $env_file
else
    echo "Invalid option"
    exit 1
fi