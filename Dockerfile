# Use official Python 3.11 image (can't upgrade to trixie, due to torio depending on ffmpeg6 not 7)
FROM python:3.11-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    git-lfs \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Git LFS setup
RUN git lfs install

# Install PlayDiffusion (To refresh --no-cache to build)
RUN git clone https://github.com/coezbek/PlayDiffusion /app/PlayDiffusion
# For local development, uncomment the following line and comment the above line
# COPY . /app/PlayDiffusion

# Set working dir inside repo
WORKDIR /app/PlayDiffusion

# Upgrade pip and install dependencies (including demo)
RUN pip install uv && uv sync

# Create HuggingFace cache mount path
ENV HF_HOME=/app/.cache/huggingface

# Expose default gradio port
EXPOSE 7860

# Set default run command
CMD ["uv", "run", "demo/gradio-demo.py"]
