# ====================================
# Multi-stage Dockerfile for Track Status Project with Miniconda
# ====================================

# Stage 1: Base image with CUDA support and Miniconda
FROM nvidia/cuda:12.1.0-dev-runtime-ubuntu22.04 AS base

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
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh -O ~/anaconda.sh && \
	/bin/bash ~/anaconda.sh -b -p /opt/conda && \
	rm ~/anaconda.sh && \
	/opt/conda/bin/conda clean -tipsy && \
	ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
	echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
	echo "conda activate base" >> ~/.bashrc && \
	find /opt/conda/ -follow -type f -name '*.a' -delete && \
	find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
	/opt/conda/bin/conda clean -afy

ENV PATH /opt/conda/bin:$PATH
COPY requirements.txt /tmp/
RUN conda update conda -y && conda create -n env10 python=3.10 -y

RUN echo "source activate env10" > ~/.bashrc
ENV PATH /opt/conda/envs/env10/bin:$PATH
# Upgrade pip in py310 environment
RUN /bin/bash -c "source activate env10"

RUN pip install --upgrade pip setuptools wheel
# Install CUDA 12.1 + cuDNN 9 ONLY in conda
# Install cuDNN 9 manually into env10
RUN /bin/bash -c "\
    wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.0.0_cuda12-archive.tar.xz -O /tmp/cudnn.tar.xz && \
    mkdir -p /tmp/cudnn_extract && \
    tar -xvf /tmp/cudnn.tar.xz -C /tmp/cudnn_extract --strip-components=1 && \
    cp -P /tmp/cudnn_extract/lib/libcudnn* /opt/conda/envs/env10/lib/ && \
    cp -r /tmp/cudnn_extract/include/* /opt/conda/envs/env10/include/ && \
    rm -rf /tmp/cudnn*"
# ====================================
# Stage 2: Dependencies installation
# ====================================
FROM base AS dependencies

# Set working directory
WORKDIR /app

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

# Set CUDA library paths
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