# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS egr-cudnn

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    apache2-utils \
    bc \
    binutils \
    bzip2 \
    cmake \
    curl \
    emacs-nox \
    gcc g++ \
    gcc-12 g++-12 \
    git-core \
    gnupg2 \
    htop \
    iproute2 \
    locales \
    net-tools \
    nginx \
    openssh-server \
    pciutils \
    python3.11 \
    python3-dev \
    software-properties-common \
    sysvinit-utils \
    sudo \
    tmux \
    tzdata \
    wget \
    zsh

RUN locale-gen en_US.UTF-8 && dpkg-reconfigure locales

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

COPY ./install-3rd-party.sh /tmp/
COPY ./gaston.patch /tmp/
RUN bash ./tmp/install-3rd-party.sh

ENV HOSTNAME="egr-cr"
RUN echo $HOSTNAME > /etc/hostname

RUN adduser --disabled-password --gecos '' egr
RUN adduser egr sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER egr
WORKDIR /home/egr
