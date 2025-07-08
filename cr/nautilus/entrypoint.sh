#!/bin/bash

set -euxo pipefail

input_file=$(basename $INPUT_FILE)
output_dir=/output

rclone copy --filter="+ ${input_file}" --filter="- *" ceph-s3:input /input

/home/egr/egr/.venv/bin/python -m workflows.run -c $INPUT_FILE

cd $output_dir
experiment_id=$(cat .experiment_id)
tarfile=${experiment_id}.tar.xz
tar cJf ${tarfile} ${experiment_id}
rclone copy --filter="+ ${tarfile}" --filter="- *" . ceph-s3:output
