# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04 AS job

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

ENV HOSTNAME="egr"
RUN echo "egr" > /etc/hostname

RUN adduser --disabled-password --gecos '' egr
RUN adduser egr sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
COPY . /home/egr/egr
RUN chown -R egr:egr /home/egr/egr

USER egr
WORKDIR /home/egr/egr
ENV PYTHONPATH=/home/egr/egr
RUN uv sync --frozen

CMD ["/bin/bash", "/home/egr/egr/cr/default/entrypoint.sh"]