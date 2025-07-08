#!/bin/bash

set -xeuo pipefail

generate_report() {
    input_file=$1
    output_dir=$2

    if [ -z $input_file ] || [ -z $output_dir ]; then
        echo "invalid i/o args input_file=$input_file, output_dir=$output_dir"
        exit 1
    fi

    if [ ! -f $input_file ]; then
        echo "No file $input_file"
        exit 1
    fi

    jupyter nbconvert \
        --config nb/configs/latex_pdf.py \
        --template nb/configs/latex_pdf.tplx \
        --to latex \
        --no-input \
        $input_file \
        --output-dir $output_dir
}

generate_report $1 $2
