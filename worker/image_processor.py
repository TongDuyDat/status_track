"""
Worker process: Lấy task từ Redis queue và xử lý pipeline (ASYNC MODE)
"""

import asyncio
import sys
import os
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import cv2
import psutil
import torch
import gc
from utils.redis_utils_async import pop_task_async, save_result_async
from pipelines.pipeline_async import pipeline_async
from core.logger import logger

# Memory thresholds
RAM_THRESHOLD = 0.85
GPU_THRESHOLD = 0.90

# Worker configuration
WORKER_BATCH_SIZE = int(os.getenv("WORKER_BATCH_SIZE", 16))
BATCH_TIMEOUT = float(os.getenv("BATCH_TIMEOUT", 0.1))  # 100ms

def check_memory():
    """Kiểm tra memory usage"""
    ram_usage = psutil.virtual_memory().percent / 100.0
    gpu_usage = 0.0
    if torch.cuda.is_available():
        try:
            gpu_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        except:
            pass
    return ram_usage > RAM_THRESHOLD or gpu_usage > GPU_THRESHOLD

def force_gc():
    """Force garbage collection"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(f"🧹 Memory cleaned - RAM: {psutil.virtual_memory().percent:.1f}%")

async def collect_batch(max_size: int, timeout: float):
    """Thu thập batch tasks từ Redis - ASYNC"""
    tasks = []
    start_time = time.time()
    
    while len(tasks) < max_size:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            break
        
        # ✅ Async Redis pop
        remaining_timeout = timeout - elapsed
        task = await pop_task_async(timeout=min(remaining_timeout, 0.01))
        
        if task:
            tasks.append(task)
        else:
            if tasks:  # Có ít nhất 1 task → return
                break
            await asyncio.sleep(0.001)
    
    return tasks

async def process_batch(tasks):
    """Xử lý batch tasks cùng lúc"""
    if not tasks:
        return
    
    batch_size = len(tasks)
    logger.info(f"🔄 Processing batch of {batch_size} tasks")
    start_time = time.time()
    
    # Check memory
    if check_memory():
        logger.warning("⚠️  Memory high, cleaning up...")
        force_gc()
        await asyncio.sleep(0.1)
    
    # Decode all images
    images = []
    valid_tasks = []
    
    decode_start = time.time()
    for task in tasks:
        try:
            image_bytes = bytes.fromhex(task["image_bytes"])
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is not None:
                images.append(image)
                valid_tasks.append(task)
        except Exception as e:
            logger.error(f"❌ Failed to decode task: {e}")
    
    decode_time = time.time() - decode_start
    
    if not images:
        return
    
    # ✅ Batch inference
    logger.info(f"🚀 Running pipeline for {len(images)} images")
    pipeline_start = time.time()
    results = await pipeline_async(images)
    pipeline_time = time.time() - pipeline_start
    
    # ✅ Save results async
    save_start = time.time()
    save_tasks = []
    for task, result in zip(valid_tasks, results):
        save_tasks.append(
            save_result_async(
                task["task_id"],
                result if result else {"error": "No result"}
            )
        )
    await asyncio.gather(*save_tasks)
    save_time = time.time() - save_start
    
    total_time = time.time() - start_time
    throughput = len(images) / pipeline_time if pipeline_time > 0 else 0
    
    logger.info(f"📊 Batch summary:")
    logger.info(f"   Total: {total_time:.3f}s")
    logger.info(f"   Decode: {decode_time:.3f}s")
    logger.info(f"   Pipeline: {pipeline_time:.3f}s ({throughput:.1f} img/s)")
    logger.info(f"   Save: {save_time:.3f}s")
    
    # Cleanup
    del images, results

async def worker_loop():
    """Main worker loop với batch processing"""
    logger.info("🚀 Worker started (ASYNC BATCH MODE)")
    logger.info(f"   Batch size: {WORKER_BATCH_SIZE}")
    logger.info(f"   Batch timeout: {BATCH_TIMEOUT}s")

    task_count = 0
    last_gc_time = time.time()

    while True:
        try:
            # ✅ Collect batch (async)
            tasks = await collect_batch(WORKER_BATCH_SIZE, BATCH_TIMEOUT)
            
            if tasks:
                await process_batch(tasks)
                task_count += len(tasks)

                # Periodic GC
                if task_count % 100 == 0 or (time.time() - last_gc_time) > 300:
                    force_gc()
                    last_gc_time = time.time()
            else:
                await asyncio.sleep(0.05)

        except KeyboardInterrupt:
            logger.info("⏹️  Worker stopped by user")
            break
        except Exception as e:
            logger.error(f"Worker error: {str(e)}")
            await asyncio.sleep(1)

def main():
    """Entry point"""
    try:
        asyncio.run(worker_loop())
    except KeyboardInterrupt:
        print("\n👋 Shutting down worker...")

if __name__ == "__main__":
    main()