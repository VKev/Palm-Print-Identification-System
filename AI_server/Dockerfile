# # Use PyTorch with CUDA and cuDNN
# FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

# RUN pip install --no-cache-dir tensorboardX causal-conv1d==1.4.0 mamba-ssm==2.2.2 timm==1.0.9 einops transformers opencv-python scipy flask python-dotenv pymongo matplotlib

# # Set working directory and copy application code
# WORKDIR /app
# # COPY . /app




FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.8

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    torch==1.8.1+cu111 \
    torchvision==0.9.1+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /app