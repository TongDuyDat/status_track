# ====================================
# Multi-stage Dockerfile for Track Status Project with Miniconda
# ====================================

# Stage 1: Base image with CUDA support and Miniconda
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/conda/bin:$PATH \
    CONDA_DIR=/opt/conda

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean --all -y && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# Configure conda to use conda-forge (fix ToS issue)
RUN /opt/conda/bin/conda config --set channel_priority strict && \
    /opt/conda/bin/conda config --prepend channels conda-forge && \
    /opt/conda/bin/conda config --set always_yes yes

# Create Python 3.10 environment
RUN /opt/conda/bin/conda create -n py310 python=3.10 -y && \
    /opt/conda/bin/conda clean --all -y && \
    echo "conda activate py310" >> ~/.bashrc

# Activate py310 environment by default
ENV PATH=/opt/conda/envs/py310/bin:$PATH \
    CONDA_DEFAULT_ENV=py310

# Upgrade pip in py310 environment
RUN pip install --upgrade pip setuptools wheel

# ====================================
# Stage 2: Dependencies installation
# ====================================
FROM base AS dependencies

# Set working directory
WORKDIR /app

# Ensure py310 environment is active
ENV PATH=/opt/conda/envs/py310/bin:$PATH \
    CONDA_DEFAULT_ENV=py310

# Copy requirements file
COPY requirements-docker.txt requirements.txt
COPY .env .env

# Install Python dependencies in layers for better caching
RUN pip install --no-cache-dir \
    fastapi==0.115.6 \
    uvicorn[standard]==0.34.0 \
    python-dotenv==1.0.1 \
    redis==6.4.0 \
    aioredis==2.0.1

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    opencv-python-headless==4.10.0.84 \
    pillow==10.4.0

RUN pip install --no-cache-dir \
    onnxruntime-gpu==1.19.0 \
    torch==2.3.1 \
    torchvision==0.18.1

RUN pip install --no-cache-dir \
    ultralytics==8.3.203 \
    psutil==7.0.0 \
    loguru==0.7.3 \
    pyyaml==6.0.2

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# ====================================
# Stage 3: Final runtime image
# ====================================
FROM base AS runtime

# Set working directory
WORKDIR /app

# Activate py310 environment
ENV PATH=/opt/conda/envs/py310/bin:$PATH \
    CONDA_DEFAULT_ENV=py310

# Set CUDA library paths (CHỈ CẦN 1 LẦN Ở ĐÂY)
ENV LD_LIBRARY_PATH="\
/usr/local/cuda/lib64:\
/usr/local/cuda/targets/x86_64-linux/lib:\
/usr/lib/x86_64-linux-gnu:\
/opt/conda/envs/py310/lib:\
${LD_LIBRARY_PATH}"

# Copy conda environment from dependencies stage
COPY --from=dependencies /opt/conda /opt/conda

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs results images images_test trt_cache

# Set permissions
RUN chmod +x start_worker_optimized.py main.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import redis; r=redis.Redis(host='${REDIS_HOST:-localhost}', port=${REDIS_PORT:-6379}); r.ping()" || exit 1

# Expose port for API
EXPOSE 8000

# Default command - optimized worker
CMD ["python", "start_worker_optimized.py"]