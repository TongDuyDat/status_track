# ====================================
# Multi-stage Dockerfile for Track Status Project
# ====================================

# Stage 1: Base image with CUDA support
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# ====================================
# Stage 2: Dependencies installation
# ====================================
FROM base AS dependencies

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements-docker.txt requirements.txt

# Install Python dependencies
# Split into multiple layers for better caching
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
RUN pip install --no-cache-dir -r requirements.txt
# ====================================
# Stage 3: Final runtime image
# ====================================
FROM base AS runtime

# Set working directory
WORKDIR /app

# Copy installed packages from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs results images images_test trt_cache

# Set permissions
RUN chmod +x start_worker_optimized.py main.py

# Expose port for API (if needed)
EXPOSE 8000

# Default command - optimized worker
CMD ["python", "start_worker_optimized.py"]