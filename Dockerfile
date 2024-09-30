# Use the base image with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1

# Use bash shell for RUN commands
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Set working directory
WORKDIR /

# Upgrade apt packages and install required dependencies
RUN apt update && \
    apt upgrade -y && \
    apt install -y \
      python3-dev \
      python3-pip \
      fonts-dejavu-core \
      rsync \
      git \
      jq \
      moreutils \
      aria2 \
      wget \
      curl \
      libglib2.0-0 \
      libsm6 \
      libgl1 \
      libxrender1 \
      libxext6 \
      ffmpeg \
      libgoogle-perftools4 \
      libtcmalloc-minimal4 \
      procps && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean -y

# Set Python alternative
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy necessary files into the image
COPY start.sh rp_handler.py ./
COPY schemas /schemas
COPY workflows /workflows

# Make the start script executable
RUN chmod +x /start.sh

# Define the command to run the container
ENTRYPOINT ["/start.sh"]
