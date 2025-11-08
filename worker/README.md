# Worker Process - License Plate Recognition

## 🚀 Khởi động Worker

### 1. Cài đặt dependencies
```bash
pip install redis numpy opencv-python
```

### 2. Cấu hình Redis
Tạo file `.env` trong thư mục gốc:
```env
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
DEBUG_MODE=false
```

### 3. Chạy Worker
```bash
# Windows PowerShell
python -m worker.image_processor

# Hoặc
python worker\image_processor.py
```

### 4. Chạy nhiều Workers song song (tăng throughput)
```bash
# Terminal 1
python -m worker.image_processor

# Terminal 2
python -m worker.image_processor

# Terminal 3
python -m worker.image_processor
```

## 📊 Monitoring

Worker sẽ log các thông tin:
- ✅ Task completed: Xử lý thành công
- ❌ Task failed: Lỗi xử lý
- 🔄 Processing task: Đang xử lý

## 🔧 Tối ưu

### Tăng số lượng workers
- **Low traffic** (< 50 req/s): 1-2 workers
- **Medium traffic** (50-100 req/s): 3-5 workers
- **High traffic** (> 100 req/s): 5-10 workers

### GPU allocation
- Nếu có nhiều GPUs, set `CUDA_VISIBLE_DEVICES`:
  ```bash
  # Worker 1: GPU 0
  set CUDA_VISIBLE_DEVICES=0
  python -m worker.image_processor
  
  # Worker 2: GPU 1
  set CUDA_VISIBLE_DEVICES=1
  python -m worker.image_processor
  ```

## 🐛 Debug Mode

Enable debug mode để lưu các crop images:
```env
DEBUG_MODE=true
```

Các ảnh crop sẽ được lưu với format: `{uuid}_{text}.jpg`
