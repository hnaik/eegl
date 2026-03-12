#!/usr/bin/env bash

gaston_bin_path=${HOME}/ws/subgraphMining/gaston-1.1/gaston
sudo cp ${gaston_bin_path} /usr/local/bin/

if [ -f "./.venv/bin/activate" ]; then
    . "./.venv/bin/activate"
fi

/bin/sleep infinity
