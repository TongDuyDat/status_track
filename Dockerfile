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
# INSTALL MAMBAFORGE (NO TOS)
# ===========================================
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

RUN pip install --upgrade pip setuptools wheel
# Install CUDA 12.1 + cuDNN 9 ONLY in conda
RUN /bin/bash -c "conda install -y -c nvidia cudnn=9.1.0 cuda-runtime=12.1"
# ====================================
# Stage 2: Dependencies installation
# ====================================
FROM base AS dependencies

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
