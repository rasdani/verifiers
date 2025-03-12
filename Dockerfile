ARG CUDA_VERSION="12.1.0"
ARG UBUNTU_VERSION="22.04"

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${UBUNTU_VERSION} AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    git-lfs \
    build-essential \
    ninja-build \
    libaio-dev \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Git LFS
RUN git lfs install


# Set working directory
WORKDIR /workspace

# Install Python 3.11
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.11 python3.11-dev python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

# Install UV package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Ensure the installed binary is on PATH
ENV PATH="/root/.local/bin:$PATH"

# Create and activate virtual environment
RUN uv venv -p python3.11 /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy project files first to use requirements
COPY --exclude=outputs/ . /workspace/verifiers/
WORKDIR /workspace/verifiers

# Install dependencies using uv as per provided commands
RUN uv pip install setuptools psutil && \
    uv pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    uv pip install -e . --no-build-isolation && \
    uv pip install flash_attn --no-build-isolation && \
    uv pip install wandb

# Create cache directory for huggingface
RUN mkdir -p /root/.cache/huggingface

# Set default command
CMD ["/bin/bash"]