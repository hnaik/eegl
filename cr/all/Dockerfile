# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04 AS egr-base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    binutils \
    cmake \
    curl \
    g++ \
    git-core \
    libboost-program-options-dev  \
    libboost-container-dev \
    locales \
    ninja-build \
    python3 \
    python3-dev \
    sysvinit-utils \
    sudo \
    tzdata \
    xz-utils


RUN locale-gen en_US.UTF-8 && dpkg-reconfigure locales

RUN mkdir external && \
    cd /external && \
    git clone https://github.com/ciaranm/glasgow-subgraph-solver.git && \
    mkdir glasgow-subgraph-solver/build && \
    cd glasgow-subgraph-solver/build && \
    cmake -G Ninja .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCATCH_INSTALL_DOCS=OFF \
        -DCATCH_INSTALL_EXTRAS=OFF \
        -DBUILD_TESTING=OFF \
        -DCATCH_CONFIG_DISABLE=ON && \
    ninja && \
    cp glasgow_subgraph_solver /external && \
    cd /external && \
    rm -rf glasgow-subgraph-solver
ENV EEGL_SOLVER_PATH=/external/glasgow_subgraph_solver

COPY . /home/egr/egr
ENV PYTHONPATH=/home/egr/egr

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

RUN apt-get remove -y \
    cmake curl g++ git-core ninja-build && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

FROM egr-base AS egr-compaas

RUN addgroup --gid 4126 egr && \
    adduser --disabled-password --gecos '' --uid 4126 --gid 4126 egr && \
    adduser egr sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R egr:egr /home/egr

USER egr
WORKDIR /home/egr/egr
ENV HOSTNAME=egr-compaas

RUN uv sync --frozen

CMD ["/bin/bash", "/home/egr/egr/cr/compaas/entrypoint.sh"]

FROM egr-base AS egr

RUN adduser --disabled-password --gecos '' egr && \
    adduser egr sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R egr:egr /home/egr

USER egr
WORKDIR /home/egr/egr
ENV HOSTNAME=egr

RUN uv sync --frozen

CMD ["/bin/bash", "/home/egr/egr/cr/default/entrypoint.sh"]

FROM egr-base AS egr-nautilus

RUN apt-get update -y -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    rclone

RUN adduser --disabled-password --gecos '' egr && \
    adduser egr sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R egr:egr /home/egr

USER egr
WORKDIR /home/egr/egr
ENV HOSTNAME=egr-nautilus

RUN uv sync --frozen

CMD ["/bin/bash", "/home/egr/egr/cr/nautilus/entrypoint.sh"]