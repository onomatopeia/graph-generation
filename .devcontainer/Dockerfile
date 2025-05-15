# Base image with CUDA
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12 and tools
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y software-properties-common wget curl gnupg2 lsb-release && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y \
        python3.12 python3.12-dev python3.12-venv && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/local/bin/pip /usr/bin/pip && \
    python -m pip install --upgrade pip && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ARG USERNAME=vscode
RUN useradd -ms /bin/bash $USERNAME
USER $USERNAME
WORKDIR /workspaces/graph-generation
