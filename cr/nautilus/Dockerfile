# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

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
    rclone \
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

COPY . /home/egr/egr
ENV PYTHONPATH=/home/egr/egr

RUN curl -Ls micro.mamba.pm/install.sh | bash -s -- && \
    mv /root/.local/bin/micromamba /usr/local/bin/micromamba && \
    mkdir /usr/local/pkgs

ENV MAMBA_ROOT_PREFIX=/usr/local/pkgs

RUN apt-get remove -y \
    cmake curl g++ git-core ninja-build && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

RUN echo "egr-nautilus" > /etc/hostname
ENV HOSTNAME="egr-nautilus"

RUN adduser --disabled-password --gecos '' egr && \
    adduser egr sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R egr:egr /home/egr

USER egr
WORKDIR /home/egr/egr

RUN micromamba env create -p ./.venv -f cr/nautilus/environment.yml

CMD ["/bin/bash", "/home/egr/egr/cr/nautilus/entrypoint.sh"]
