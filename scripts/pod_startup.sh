#!/usr/bin/env bash

gaston_bin_path=${HOME}/ws/subgraphMining/gaston-1.1/gaston
sudo cp ${gaston_bin_path} /usr/local/bin/

# eval "$(mamba shell.zsh hook)"
if [ -f "/usr/local/etc/profile.d/mamba.sh" ]; then
    . "/usr/local/etc/profile.d/mamba.sh"
fi
mamba activate ./.venv

/bin/sleep infinity
