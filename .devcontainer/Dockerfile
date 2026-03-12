# syntax=docker/dockerfile:1
FROM docker.io/nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    binutils \
    cmake \
    curl \
    gcc g++ \
    git-core \
    libboost-program-options-dev  \
    libboost-container-dev \
    libgmp-dev \
    locales \
    make \
    ninja-build \
    python3 \
    python3-dev \
    sudo \
    sysvinit-utils \
    tzdata

RUN locale-gen en_US.UTF-8 && dpkg-reconfigure locales

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

RUN mkdir external && \
    cd /external && \
    git clone https://github.com/ciaranm/glasgow-subgraph-solver.git && \
    mkdir glasgow-subgraph-solver/build && \
    cd glasgow-subgraph-solver/build && \
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Release .. && \
    ninja && \
    cp glasgow_subgraph_solver /external && \
    cd /external && \
    rm -rf glasgow-subgraph-solver
ENV EEGL_SOLVER_PATH=/external/glasgow_subgraph_solver

RUN apt-get remove -y \
    cmake curl gcc g++ git-core ninja-build && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

ARG UID=1000
ARG GID=1000

ENV HOSTNAME="eegl"
RUN echo "eegl" > /etc/hostname

RUN set -e; \
    existing_group=$(getent group ${GID} | cut -d: -f1 || true); \
    if [ -n "$existing_group" ]; then groupmod -n eegl "$existing_group"; \
    else groupadd --gid ${GID} eegl; fi; \
    existing_user=$(getent passwd ${UID} | cut -d: -f1 || true); \
    if [ -n "$existing_user" ]; then usermod -l eegl -d /home/eegl -m "$existing_user"; \
    else useradd --uid ${UID} --gid ${GID} --create-home --shell /bin/bash eegl; fi; \
    adduser eegl sudo; \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers; \
    mkdir -p /home/eegl/eegl; \
    chown eegl:eegl /home/eegl/eegl

USER eegl
WORKDIR /home/eegl/eegl
ENV PYTHONPATH=/home/eegl/eegl