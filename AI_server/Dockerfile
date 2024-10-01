# Use PyTorch with CUDA and cuDNN
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 libgstreamer-1.0-0 libgstreamer-plugins-base1.0-0
RUN pip install --no-cache-dir tensorboardX causal-conv1d==1.4.0 mamba-ssm==2.2.2 timm==1.0.9 einops transformers opencv-python scipy flask python-dotenv pymongo matplotlib

# Set working directory and copy application code
WORKDIR /app
COPY . /app

