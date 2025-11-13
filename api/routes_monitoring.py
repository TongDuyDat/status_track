"""
Monitoring endpoints để theo dõi hệ thống
"""

from fastapi import APIRouter
import psutil
import torch
import os

PIPELINE_MODE = os.getenv("PIPELINE_MODE", "staged").lower()

router = APIRouter(prefix="/api/monitor", tags=["Monitoring"])


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "License Plate Recognition"}


@router.get("/memory")
async def get_memory_stats():
    """Lấy thông tin memory usage"""
    # RAM
    ram = psutil.virtual_memory()
    ram_stats = {
        "total_gb": round(ram.total / (1024**3), 2),
        "used_gb": round(ram.used / (1024**3), 2),
        "available_gb": round(ram.available / (1024**3), 2),
        "percent": ram.percent,
    }

    # GPU
    gpu_stats = {"available": False}
    if torch.cuda.is_available():
        try:
            gpu_stats = {
                "available": True,
                "device_name": torch.cuda.get_device_name(0),
                "allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 2),
                "reserved_gb": round(torch.cuda.memory_reserved() / (1024**3), 2),
                "max_allocated_gb": round(
                    torch.cuda.max_memory_allocated() / (1024**3), 2
                ),
            }
        except:
            pass

    return {
        "ram": ram_stats,
        "gpu": gpu_stats,
    }


# @router.get("/batch-managers")
# async def get_batch_manager_stats():
#     """Lấy thống kê các BatchManager"""
#     stats = {
#         "truck_detector": truck_batcher.get_stats(),
#         "text_detector": text_batcher.get_stats(),
#         "ocr": ocr_batcher.get_stats(),
#     }

#     # Add pipeline scheduler stats if in staged mode
#     if PIPELINE_MODE == "staged":
#         try:
#             scheduler = get_pipeline_scheduler()
#             stats["pipeline_scheduler"] = scheduler.get_stats()
#         except:
#             pass

#     return stats


@router.get("/system")
async def get_system_stats():
    """Lấy thông tin tổng quan hệ thống"""
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()

    # Disk
    disk = psutil.disk_usage("/")
    disk_stats = {
        "total_gb": round(disk.total / (1024**3), 2),
        "used_gb": round(disk.used / (1024**3), 2),
        "free_gb": round(disk.free / (1024**3), 2),
        "percent": disk.percent,
    }

    return {
        "cpu": {
            "count": cpu_count,
            "percent": cpu_percent,
        },
        "disk": disk_stats,
    }


@router.post("/gc")
async def force_garbage_collection():
    """Trigger garbage collection thủ công"""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"status": "success", "message": "Garbage collection completed"}
