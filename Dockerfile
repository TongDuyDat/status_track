# ====================================
# Multi-stage Dockerfile for Track Status Project with Miniconda
# (Fixed: consistent env name env10, base image without cuDNN8, LD_LIBRARY_PATH priority)
# ====================================

# Stage 1: Base image with CUDA toolkit (use -devel so nvcc/toolkit available)
# NOTE: choose image WITHOUT cuDNN preinstalled so your cuDNN9 copy is effective.
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/conda/bin:$PATH \
    CONDA_DIR=/opt/conda

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl bzip2 ca-certificates \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 libgl1-mesa-glx \
    git build-essential pkg-config && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda (smaller than full Anaconda; keep original if you prefer Anaconda)
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy

ENV PATH=/opt/conda/bin:$PATH

SHELL ["/bin/bash", "-c"]

# Create conda env env10 with python 3.10
RUN conda update -y conda && conda create -y -n env10 python=3.10 && conda clean -afy

# Ensure we can use `conda activate` in RUN
ENV PATH=/opt/conda/envs/env10/bin:/opt/conda/bin:$PATH

# Upgrade pip inside env10
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate env10 && \
    pip install --upgrade pip setuptools wheel

# Install cuDNN 9 manually into env10 (you used this archive)
# This copies libcudnn and headers into the conda env lib/include
RUN wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.0.0_cuda12-archive.tar.xz -O /tmp/cudnn.tar.xz && \
    mkdir -p /tmp/cudnn_extract && \
    tar -xvf /tmp/cudnn.tar.xz -C /tmp/cudnn_extract --strip-components=1 && \
    cp -P /tmp/cudnn_extract/lib/libcudnn* /opt/conda/envs/env10/lib/ && \
    cp -P /tmp/cudnn_extract/lib/libcudnn* /usr/local/cuda/lib64/ || true && \
    cp -r /tmp/cudnn_extract/include/* /opt/conda/envs/env10/include/ && \
    rm -rf /tmp/cudnn* /tmp/cudnn_extract

# Note: we also try copying to /usr/local/cuda/lib64 to help system linker if needed (optional).
# ====================================
# Stage 2: Dependencies installation
# ====================================
FROM base AS dependencies

WORKDIR /app

# Copy requirements - use your provided file
COPY requirements-docker.txt requirements.txt
COPY .env .env

# Install Python dependencies INSIDE env10 to ensure correct placement
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate env10 && \
    pip install --no-cache-dir \
        fastapi==0.115.6 uvicorn[standard]==0.34.0 python-dotenv==1.0.1 redis==6.4.0 aioredis==2.0.1 && \
    pip install --no-cache-dir \
        numpy==1.26.4 opencv-python-headless==4.10.0.84 pillow==10.4.0 && \
    pip install --no-cache-dir \
        onnxruntime-gpu==1.19.0 torch==2.3.1 torchvision==0.18.1 && \
    pip install --no-cache-dir \
        ultralytics==8.3.203 psutil==7.0.0 loguru==0.7.3 pyyaml==6.0.2 && \
    pip install --no-cache-dir -r requirements.txt

# Clean pip cache optionally
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate env10 && pip cache purge || true

# ====================================
# Stage 3: Final runtime image
# ====================================
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS runtime

# Copy conda (including env10) from dependencies stage
COPY --from=dependencies /opt/conda /opt/conda

# Ensure runtime sees nvcc and conda env, and make env10's lib first in LD_LIBRARY_PATH
ENV PATH=/opt/conda/envs/env10/bin:/opt/conda/bin:/usr/local/cuda/bin:$PATH \
    CONDA_DEFAULT_ENV=env10 \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/opt/conda/envs/env10/lib:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}

WORKDIR /app

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs results images images_test trt_cache

# Make scripts executable
RUN chmod +x start_worker_optimized.py main.py || true

# Simple healthcheck (adjust if redis not installed)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import redis; r=redis.Redis(host='${REDIS_HOST:-localhost}', port=${REDIS_PORT:-6379}); r.ping()" || exit 1

EXPOSE 8000

# Final command: activate env10 then run
CMD ["/bin/bash", "-lc", "source /opt/conda/etc/profile.d/conda.sh && conda activate env10 && python start_worker_optimized.py"]
