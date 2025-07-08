#!/bin/bash

set -euo pipefail

ROOT_DIR=$(realpath $(dirname $0)/..)
SFW=gaston-1.1

build_dir=/tmp/${SFW}
rm -rf ${build_dir}
wget -c https://liacs.leidenuniv.nl/~nijssensgr/gaston/${SFW}.tar.gz -O - | \
    tar xzf - -C /tmp
pushd ${build_dir}
patch main.cpp < /tmp/gaston.patch
make 
sudo cp gaston /usr/local/bin/
popd
