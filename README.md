# track_status

Hướng dẫn cài đặt và sử dụng (tiếng Việt)

## Mục đích
Project này chứa một pipeline xử lý ảnh/ocr được tối ưu cho GPU với chế độ "staged pipeline" (pipelined stages) và các worker xử lý bất đồng bộ. Bao gồm cả API (FastAPI) để upload/monitoring, các pipeline xử lý văn bản/nhận diện, và scripts để khởi động worker tối ưu.

## Các cách cài đặt

Project hỗ trợ 2 cách cài đặt:

### 🐳 Cách 1: Docker (Khuyến nghị - Dễ nhất)
- ✅ Không cần cài đặt Python, CUDA, dependencies
- ✅ Tự động cài Redis
- ✅ Môi trường chuẩn, reproducible
- ✅ Dễ deploy và scale
- 👉 [Xem hướng dẫn Docker](#cài-đặt-với-docker)

### 🐍 Cách 2: Conda (Development)
- ✅ Phù hợp cho development và debug
- ✅ Có thể custom dependencies
- ⚠️ Cần cài CUDA, Redis riêng
- 👉 [Xem hướng dẫn Conda](#cài-đặt-với-conda)

---

## 🐳 Cài đặt với Docker

### Yêu cầu
- **Docker Desktop** (Windows/Mac) hoặc Docker Engine (Linux)
- **NVIDIA Docker runtime** (cho GPU support)
  - Windows: Docker Desktop với WSL2 + NVIDIA Container Toolkit
  - Linux: nvidia-docker2

### Cài đặt NVIDIA Container Toolkit (cho GPU)

#### Windows với WSL2:
```powershell
# Trong WSL2 Ubuntu
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

#### Linux:
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Khởi động project với Docker Compose

```powershell
# 1. Clone repository (nếu chưa có)
git clone <repository-url>
cd track_status

# 2. Tạo file .env từ template
Copy-Item .env.example .env

# 3. Build và khởi động tất cả services (Redis + Worker)
docker-compose up -d

# 4. Xem logs
docker-compose logs -f worker

# 5. Kiểm tra status
docker-compose ps
```

### Các lệnh Docker hữu ích

```powershell
# Dừng tất cả services
docker-compose down

# Rebuild image khi có thay đổi code
docker-compose up -d --build

# Xem logs của worker
docker-compose logs -f worker

# Xem logs của Redis
docker-compose logs -f redis

# Chạy chỉ Redis
docker-compose up -d redis

# Chạy chỉ Worker (yêu cầu Redis đã chạy)
docker-compose up -d worker

# Chạy API service
docker-compose up -d api

# Vào terminal của worker container
docker-compose exec worker bash

# Kiểm tra GPU trong container
docker-compose exec worker nvidia-smi

# Restart worker
docker-compose restart worker

# Xóa tất cả (bao gồm volumes)
docker-compose down -v
```

### Cấu hình Docker

Chỉnh sửa file `docker-compose.yml` hoặc `.env` để thay đổi cấu hình:

```env
# .env file
REDIS_PORT=6379
API_PORT=8000
WORKER_BATCH_SIZE=16
LOG_LEVEL=INFO
```

### Monitor GPU trong Docker

```powershell
# Xem GPU usage từ host
nvidia-smi

# Xem GPU usage trong container
docker-compose exec worker python monitor_gpu.py
```

---

## 🐍 Cài đặt với Conda

### Yêu cầu cơ bản
- **Anaconda/Miniconda**: khuyến nghị để quản lý môi trường Python
- **Python 3.8+** (khuyến nghị 3.9/3.10)
- **GPU + CUDA** nếu muốn chạy ONNX / GPU-accelerated runtime
- **Redis Server**: để quản lý task queue và lưu kết quả
- Một số thư viện thường thấy trong project: FastAPI, uvicorn, onnxruntime (hoặc onnxruntime-gpu), numpy, redis, python-dotenv, asyncio, aiohttp

## Cài đặt Redis trên Windows

### Cách 1: Dùng WSL2 (khuyến nghị)
```powershell
# Trong WSL2 Ubuntu
sudo apt update
sudo apt install redis-server

# Khởi động Redis
sudo service redis-server start

# Kiểm tra Redis đang chạy
redis-cli ping
# Kết quả: PONG
```

### Cách 2: Dùng Redis for Windows (community port)
1. Tải Redis for Windows từ: https://github.com/tporadowski/redis/releases
2. Giải nén và chạy `redis-server.exe`
3. Kiểm tra bằng `redis-cli.exe ping`

### Cách 3: Dùng Docker (dễ nhất)
```powershell
# Pull và chạy Redis container
docker run -d -p 6379:6379 --name redis redis:latest

# Kiểm tra
docker exec -it redis redis-cli ping
# Kết quả: PONG
```

Sau khi cài đặt, Redis sẽ chạy tại `localhost:6379` (mặc định).

## Chuẩn bị môi trường Python với Conda (Windows PowerShell)

### 1. Tạo conda environment mới
```powershell
# Tạo environment với Python 3.10
conda create -n track_status python=3.10 -y

# Kích hoạt environment
conda activate track_status
```

### 2. Cài đặt phụ thuộc cơ bản
Nếu repository có `requirements.txt`:

```powershell
pip install -r requirements.txt
```

Nếu không có `requirements.txt`, cài các package cơ bản:

```powershell
# Web framework & async
pip install fastapi uvicorn python-dotenv

# Redis client
pip install redis aioredis

# Data processing
pip install numpy pillow

# ONNX Runtime (CPU version)
pip install onnxruntime

# Hoặc GPU version (cần CUDA đã cài)
pip install onnxruntime-gpu
```

### 3. Cài đặt CUDA & cuDNN (nếu dùng GPU)
- Cài CUDA Toolkit phù hợp với phiên bản onnxruntime-gpu
- Tham khảo: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html

Lưu ý: Nếu muốn sử dụng GPU với onnxruntime, cài `onnxruntime-gpu` thay cho `onnxruntime` và đảm bảo driver/CUDA tương thích.

## Biến môi trường quan trọng

### Tạo file `.env` tại thư mục gốc project
Copy từ file mẫu và chỉnh sửa theo nhu cầu:

```powershell
# Copy file mẫu
Copy-Item .env.example .env

# Chỉnh sửa file .env bằng text editor
notepad .env
```

Nội dung tham khảo (xem chi tiết trong `.env.example`):

```env
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# Pipeline Configuration
PIPELINE_MODE=staged
WORKER_BATCH_SIZE=16
BATCH_TIMEOUT=0.05
MAX_CONCURRENT_BATCHES=10

# Logging
LOG_LEVEL=INFO

# Memory Thresholds
RAM_THRESHOLD=0.85
GPU_THRESHOLD=0.90
MAX_QUEUE_SIZE=1000
```

### Hoặc set trực tiếp trong PowerShell (tạm thời)
```powershell
# Redis
$env:REDIS_HOST = "localhost"
$env:REDIS_PORT = "6379"
$env:REDIS_DB = "0"

# Pipeline
$env:PIPELINE_MODE = "staged"
$env:WORKER_BATCH_SIZE = "16"
$env:BATCH_TIMEOUT = "0.05"
$env:MAX_CONCURRENT_BATCHES = "10"
$env:LOG_LEVEL = "INFO"
```

### Giải thích các biến quan trọng:

**Redis:**
- `REDIS_HOST`: địa chỉ Redis server (mặc định: localhost)
- `REDIS_PORT`: port Redis (mặc định: 6379)
- `REDIS_DB`: database number (mặc định: 0)
- `REDIS_PASSWORD`: password (để trống nếu không dùng auth)

**Pipeline:**
- `PIPELINE_MODE`: `staged` để bật chế độ pipeline nhiều stage (khuyến nghị)
- `WORKER_BATCH_SIZE`: số ảnh tối đa gom vào 1 batch xử lý trên GPU (mặc định: 16)
- `BATCH_TIMEOUT`: thời gian chờ (giây) để gom batch (mặc định: 0.05 = 50ms)
- `MAX_CONCURRENT_BATCHES`: số batch chạy đồng thời (mặc định: 10)
- `LOG_LEVEL`: mức độ log `INFO`/`DEBUG`

## Chạy worker / pipeline

**Quan trọng:** Đảm bảo Redis đã chạy trước khi khởi động worker!

Kiểm tra Redis:
```powershell
# Test kết nối Redis
redis-cli ping
# Hoặc nếu dùng Docker:
docker exec -it redis redis-cli ping
```

### Khởi động worker với staged pipeline (khuyến nghị)

```powershell
# Kích hoạt conda environment
conda activate track_status

# Chạy worker (script này tự set các env variables tối ưu)
python .\start_staged_worker.py
```

Script `start_staged_worker.py` sẽ tự động:
- Set `PIPELINE_MODE=staged`
- Tăng batch size lên 16 để tối ưu GPU
- Cấu hình các thông số batch timeout và concurrent batches

### Các cách chạy khác

```powershell
# Hoặc dùng main.py
python .\main.py

# Hoặc worker optimized
python .\start_worker_optimized.py
```

Mỗi script có thể chứa cấu hình khác nhau; hãy mở file tương ứng để xem các thông số cụ thể.

### Những gì bạn nên thấy khi worker khởi động thành công:
```
✅ '🚀 Pipeline scheduler started with 3 stages'
✅ '🟢 Stage 1/2/3 worker started'
✅ '[BatchManager] Processing batch_size=16+' (không phải 1!)
✅ GPU utilization 80%+ (kiểm tra với: python monitor_gpu.py)
```

## Chạy API (FastAPI)
Project có phần API để upload ảnh/đa luồng và monitoring. Để chạy API development server (nếu file chứa `app = FastAPI(...)` nằm trong một module):

1. Tìm file định nghĩa `app = FastAPI(...)` (ví dụ `api/app.py` hoặc `api/__init__.py`).
2. Chạy uvicorn, ví dụ:

```powershell
uvicorn api.app:app --reload
```

(Lưu ý: điều chỉnh `api.app` thành module đúng chứa `app` trong dự án.)

## Monitor & kiểm tra GPU
- Có script tiện ích:

```powershell
python .\monitor_gpu.py
```

- Có tests liên quan GPU và hiệu năng trong thư mục `tests/` (ví dụ `tests/test_gpu_load.py`). Chạy test:

```powershell
python -m pytest -q tests/
```

## Kiểm thử nhanh (smoke tests)
- Dùng các test có sẵn trong `tests/` để kiểm chứng pipeline và GPU load quick tests.

## File/Thư mục quan trọng
- `main.py` — quick-start cho staged pipeline (cấu hình env và chạy worker)
- `start_staged_worker.py`, `start_worker_optimized.py` — scripts khởi động worker với các cấu hình khác nhau
- `worker/image_processor.py` — logic xử lý ảnh chính
- `pipelines/` — chứa các pipeline (text detection/recognition, tracking, v.v)
- `api/` — routes cho upload và monitoring
- `docs/` — tài liệu nội bộ, ví dụ `ONNX_GPU_FIX.md`, `GPU_OPTIMIZATION.md` (tham khảo nếu gặp lỗi GPU/ONNX)
- `tools/` — scripts tiện ích (ví dụ `monitor_memory.py`, `debug_pipeline.py`)

## Vấn đề thường gặp & gợi ý khắc phục

### Docker

#### GPU không nhận diện trong container
```powershell
# Kiểm tra nvidia-docker hoạt động
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Nếu lỗi, restart Docker
# Windows: Restart Docker Desktop
# Linux: sudo systemctl restart docker

# Kiểm tra driver và runtime
nvidia-smi
docker info | grep -i runtime
```

#### Container không khởi động
```powershell
# Xem logs chi tiết
docker-compose logs worker

# Rebuild image
docker-compose build --no-cache worker
docker-compose up -d worker

# Kiểm tra Redis đã chạy chưa
docker-compose ps redis
docker-compose exec redis redis-cli ping
```

#### Out of memory trong container
```powershell
# Giảm batch size trong .env hoặc docker-compose.yml
WORKER_BATCH_SIZE=8
TRUCK_MAX_BATCH=16
TEXT_MAX_BATCH=8
OCR_MAX_BATCH=32

# Restart worker
docker-compose restart worker
```

### Redis
- **Lỗi `ConnectionError: Error 10061`**: Redis chưa chạy. Khởi động Redis server trước.
- **Lỗi `WRONGPASS invalid username-password pair`**: Sai password Redis. Kiểm tra `REDIS_PASSWORD` trong `.env`.
- **Lỗi connection timeout**: Kiểm tra `REDIS_HOST` và `REDIS_PORT` có đúng không.

### Python Environment
- **Lỗi không tìm thấy module/thiếu package**: 
  - Kiểm tra conda environment đã active chưa: `conda activate track_status`
  - Cài lại dependencies: `pip install -r requirements.txt`
- **Import error**: Đảm bảo chạy từ thư mục gốc project (nơi có `main.py`)

### ONNX & GPU
- **ONNX chạy chậm hoặc gặp lỗi CUDA**: 
  - Xem `docs/ONNX_GPU_FIX.md` để biết các fix và flags khuyến nghị
  - Kiểm tra version CUDA tương thích với onnxruntime-gpu
- **GPU out-of-memory**: 
  - Giảm `WORKER_BATCH_SIZE` (thử 8 hoặc 4)
  - Bật mixed precision nếu pipeline hỗ trợ FP16
  - Kiểm tra GPU memory: `python monitor_gpu.py`

### API
- **API không khởi động**: 
  - Kiểm tra file nơi `app = FastAPI(...)` 
  - Chạy `uvicorn` với module path chính xác
  - Kiểm tra port có bị chiếm không: `netstat -ano | findstr :8000`

---

## 🚀 Quick Reference

### Docker Commands

```powershell
# Start everything
docker-compose up -d

# View logs
docker-compose logs -f worker

# Check status
docker-compose ps

# Stop everything
docker-compose down

# Rebuild after code changes
docker-compose up -d --build

# Check environment
.\check-docker-env.ps1
```

### Conda Commands

```powershell
# Activate environment
conda activate track_status

# Start worker
python .\start_staged_worker.py

# Monitor GPU
python .\monitor_gpu.py

# Run tests
python -m pytest tests/
```

## 📚 Tài liệu bổ sung

- **[DOCKER.md](DOCKER.md)** - Hướng dẫn chi tiết về Docker deployment
- **[docs/GPU_OPTIMIZATION.md](docs/GPU_OPTIMIZATION.md)** - Tối ưu GPU
- **[docs/ONNX_GPU_FIX.md](docs/ONNX_GPU_FIX.md)** - Fix ONNX GPU issues
- **[docs/STAGED_PIPELINE_OPTIMIZATION.md](docs/STAGED_PIPELINE_OPTIMIZATION.md)** - Pipeline optimization
- **[worker/README.md](worker/README.md)** - Worker documentation

## 🤝 Đóng góp

Nếu gặp vấn đề hoặc có đề xuất cải tiến:
1. Tạo issue mô tả chi tiết
2. Fork repository
3. Tạo branch cho feature/fix
4. Submit pull request

## 📄 License

[Thêm license information nếu có]

---

**Repository**: [TongDuyDat/status_track](https://github.com/TongDuyDat/status_track)

Cần hỗ trợ? Tạo issue trên GitHub hoặc liên hệ maintainer. 
