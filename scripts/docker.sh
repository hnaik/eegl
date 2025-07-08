#!/bin/bash

set -euo pipefail

command=$1

if [ $command == "build" ]; then
    pushd cr
    docker-compose build
    popd
elif [ $command == "run" ]; then
    pushd cr
    docker-compose up --build -d
    popd
elif [ $command == "login" ]; then
    docker exec -it egr-cr /bin/bash
fi
