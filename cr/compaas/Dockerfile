# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04 AS job-compaas

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    binutils \
    cmake \
    curl \
    g++ \
    git-core \
    libboost-container-dev \
    libboost-program-options-dev \
    locales \
    ninja-build \
    python3 \
    python3-dev \
    sudo \
    sysvinit-utils \
    tzdata \
    xz-utils

RUN locale-gen en_US.UTF-8 && dpkg-reconfigure locales

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

RUN addgroup --gid 4126 egr && \
    adduser --disabled-password --gecos '' --uid 4126 --gid 4126 egr && \
    adduser egr sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

ENV HOSTNAME='egr-compaas'
RUN echo $HOSTNAME > /etc/hostname

USER egr
WORKDIR /home/egr/egr
ENV PYTHONPATH=/home/egr/egr

CMD ["/bin/bash", "/home/egr/egr/cr/compaas/entrypoint.sh"]