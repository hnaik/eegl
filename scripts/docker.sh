#!/bin/bash

set -euo pipefail

command=$1

ROOT_DIR=$(realpath $(dirname $0)/..)

BUILD_ARGS="--build-arg UID=$(id -u) --build-arg GID=$(id -g)"

if command -v podman &>/dev/null; then
    DOCKER=podman
    USERNS="--userns=keep-id"
    REPLACE="--replace"
else
    DOCKER=docker
    USERNS=""
    REPLACE=""
fi

if [ $command == "build" ]; then
    ${DOCKER} build ${BUILD_ARGS} -f ${ROOT_DIR}/.devcontainer/Dockerfile -t eegl ${ROOT_DIR}
elif [ $command == "run" ]; then
    ${DOCKER} build ${BUILD_ARGS} -f ${ROOT_DIR}/.devcontainer/Dockerfile -t eegl ${ROOT_DIR}
    ${DOCKER} run -d --name eegl \
        --hostname eegl \
        --add-host=eegl:127.0.0.1 \
        --network host \
        -v ${ROOT_DIR}:/home/eegl/eegl \
        -w /home/eegl/eegl \
        --user eegl \
        ${USERNS} ${REPLACE} \
        eegl /bin/sleep infinity
elif [ $command == "login" ]; then
    ${DOCKER} exec -it eegl /bin/bash
fi
