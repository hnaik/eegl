# syntax=docker/dockerfile:1
FROM nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04

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
    locales \
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

ENV HOSTNAME="eegl"
RUN echo "eegl" > /etc/hostname

RUN adduser --disabled-password --gecos '' eegl
RUN adduser eegl sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
COPY . /home/eegl/eegl
RUN chown -R eegl:eegl /home/eegl/eegl

USER eegl
WORKDIR /home/eegl/eegl
ENV PYTHONPATH=/home/eegl/eegl