#!/bin/bash

set -euxo pipefail

output_dir=/output

/home/egr/egr/.venv/bin/python -m workflows.run -c $INPUT_FILE

cd $output_dir
experiment_id=$(cat .experiment_id)
tarfile=${experiment_id}.tar.xz
tar cJf ${tarfile} ${experiment_id}
cp $tarfile /results/
