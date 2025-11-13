# =========================================================
# Base: CUDA 12.1 + cuDNN9 + Miniconda inside Docker
# =========================================================
FROM nvcr.io/nvidia/cuda:12.1.1-cudnn9-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/conda/bin:$PATH"

# =====================================
# INSTALL SYSTEM PACKAGES
# =====================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git ca-certificates \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgl1-mesa-glx libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# =====================================
# INSTALL MINICONDA
# =====================================
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && rm miniconda.sh

SHELL ["/bin/bash", "-c"]

# =====================================
# CREATE CONDA ENV
# =====================================
RUN conda create -y -n track python=3.10 && conda clean -afy

ENV CONDA_DEFAULT_ENV=track
ENV CONDA_PREFIX=/opt/conda/envs/track
ENV PATH="$CONDA_PREFIX/bin:$PATH"

# =====================================
# FIX LD_LIBRARY_PATH (QUAN TRỌNG NHẤT)
# =====================================
ENV LD_LIBRARY_PATH="\
/usr/local/cuda/lib64:\
/usr/local/cuda/targets/x86_64-linux/lib:\
/usr/lib/x86_64-linux-gnu:\
/opt/conda/envs/track/lib:\
${LD_LIBRARY_PATH}"

# =====================================
# INSTALL PYTHON DEPENDENCIES
# =====================================
RUN conda activate track && \
    pip install numpy==1.26.4 opencv-python-headless==4.10.0.84 pillow==10.4.0 && \
    pip install fastapi uvicorn python-dotenv redis aioredis loguru pyyaml && \
    pip install onnxruntime-gpu==1.20.0 && \
    pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install ultralytics==8.3.203

# =====================================
# COPY CODE
# =====================================
WORKDIR /app
COPY . .
RUN mkdir -p logs results images images_test trt_cache

EXPOSE 8000
CMD ["python", "start_worker_optimized.py"]
