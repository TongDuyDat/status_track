# ===========================================
# Base CUDA 12.1 (no cuDNN included)
# ===========================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/conda/bin:$PATH"

# ===========================================
# System packages
# ===========================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git ca-certificates \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx libgomp1 \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# ===========================================
# ADD NVIDIA ML repo (cuDNN 9 + TensorRT 10)
# ===========================================
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/nvidia-archive-keyring.gpg] \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" \
    > /etc/apt/sources.list.d/cuda.list && \
    echo "deb [signed-by=/usr/share/keyrings/nvidia-archive-keyring.gpg] \
    https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2204/x86_64/ /" \
    > /etc/apt/sources.list.d/nvidia-ml.list

# ===========================================
# Install cuDNN 9 + TensorRT 10
# ===========================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn9 \
    libcudnn9-dev \
    tensorrt \
    tensorrt-dev \
    && rm -rf /var/lib/apt/lists/*

# ===========================================
# INSTALL MAMBAFORGE (NO TOS)
# ===========================================
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh \
    -O /tmp/mambaforge.sh && \
    bash /tmp/mambaforge.sh -b -p /opt/conda && rm /tmp/mambaforge.sh

SHELL ["/bin/bash", "-c"]

# ===========================================
# CREATE PYTHON 3.10 ENVIRONMENT
# ===========================================
RUN conda create -y -n track -c conda-forge python=3.10 && \
    conda clean --all -y

ENV CONDA_PREFIX="/opt/conda/envs/track"
ENV PATH="$CONDA_PREFIX/bin:$PATH"
ENV CONDA_DEFAULT_ENV=track

# ===========================================
# LD_LIBRARY_PATH (cuDNN 9 + TensorRT10)
# ===========================================
ENV LD_LIBRARY_PATH="\
/usr/local/cuda/lib64:\
/usr/local/cuda/targets/x86_64-linux/lib:\
/usr/lib/x86_64-linux-gnu:\
/usr/lib/x86_64-linux-gnu/tensorrt:\
/opt/conda/envs/track/lib:\
${LD_LIBRARY_PATH}"



# ===========================================
# Stage 2 — Install dependencies
# ===========================================
FROM base AS dependencies

WORKDIR /app
COPY requirements-docker.txt requirements.txt
COPY .env .env

RUN pip install --upgrade pip setuptools wheel

RUN pip install fastapi uvicorn[standard] python-dotenv redis aioredis loguru pyyaml
RUN pip install numpy==1.26.4 opencv-python-headless==4.10.0.84 pillow==10.4.0

# PyTorch for CUDA 12.1
RUN pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

# ONNX Runtime + TensorRT 10
RUN pip install onnxruntime-gpu==1.20.0

RUN pip install ultralytics==8.3.203 psutil==7.0.0
RUN pip install -r requirements.txt


# ===========================================
# Stage 3 — Runtime
# ===========================================
FROM base AS runtime

WORKDIR /app
COPY --from=dependencies /opt/conda /opt/conda
COPY . .

RUN mkdir -p logs results images images_test trt_cache
RUN chmod +x start_worker_optimized.py main.py

EXPOSE 8000
CMD ["python", "start_worker_optimized.py"]
