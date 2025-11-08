import asyncio, time
import numpy as np
import psutil
import torch
from core.logger import logger


class AdaptiveBatchManager:
    def __init__(
        self,
        infer_fn,
        min_batch=2,
        max_batch=16,
        target_latency=0.1,
        max_queue_size=1000,
        memory_threshold=0.85,
        min_wait=0.005,  # ✅ Thêm min_wait (5ms)
    ):
        self.infer_fn = infer_fn
        self.min_batch = min_batch
        self.max_batch = max_batch
        self.batch_size = min_batch
        self.target_latency = target_latency
        self.max_wait = 0.05
        self.min_wait = min_wait  # ✅ Minimum wait time
        self.queue = []
        self.lock = asyncio.Lock()
        self._started = False
        self._last_latency = target_latency
        self._last_run = time.time()

        # Memory management
        self.max_queue_size = max_queue_size
        self.memory_threshold = memory_threshold
        self._check_gpu = torch.cuda.is_available()
        
        # ✅ Stats for debugging
        self._total_batches = 0
        self._total_items = 0

    async def infer(self, data):
        if not self._started:
            asyncio.create_task(self._loop())
            self._started = True

        # ✅ Backpressure: Chờ nếu queue quá dài
        while len(self.queue) >= self.max_queue_size:
            await asyncio.sleep(0.01)

        fut = asyncio.get_event_loop().create_future()
        async with self.lock:
            self.queue.append((data, fut))
        return await fut

    async def _loop(self):
        """
        ✅ IMPROVED: Chờ đủ batch hoặc timeout
        """
        while True:
            # ✅ Check queue có items không
            async with self.lock:
                has_items = len(self.queue) > 0
            
            if not has_items:
                await asyncio.sleep(self.min_wait)
                continue
            
            # ✅ Chờ để gom batch (adaptive wait time)
            wait_start = time.time()
            while True:
                async with self.lock:
                    q_len = len(self.queue)
                
                # ✅ Điều kiện trigger batch:
                # 1. Queue đủ batch_size
                # 2. Hoặc đã chờ đủ max_wait
                elapsed = time.time() - wait_start
                if q_len >= self.batch_size or elapsed >= self.max_wait:
                    break
                
                # Chờ thêm 1ms
                await asyncio.sleep(0.001)
            
            # ✅ Lấy batch
            start = time.time()
            async with self.lock:
                if not self.queue:
                    continue
                n = min(self.batch_size, len(self.queue))
                batch = self.queue[:n]
                self.queue = self.queue[n:]
            
            inputs, futs = zip(*batch)
            
            # ✅ Logging
            # logger.debug(f"[BatchManager] Processing batch_size={len(batch)}, queue_remaining={len(self.queue)}")
            
            try:
                results = self.infer_fn(list(inputs))
                for fut, res in zip(futs, results):
                    fut.set_result(res)
            except Exception as e:
                # logger.error(f"[BatchManager] Inference failed: {e}")
                for fut in futs:
                    fut.set_exception(e)
            
            latency = time.time() - start
            
            # ✅ Update stats
            self._total_batches += 1
            self._total_items += len(batch)
            
            # ✅ Điều chỉnh params
            self._adjust_params(latency, len(batch))

    def _get_memory_usage(self):
        """Lấy memory usage (RAM + GPU)"""
        ram_percent = psutil.virtual_memory().percent / 100.0

        gpu_percent = 0.0
        if self._check_gpu:
            try:
                gpu_mem = (
                    torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                )
                gpu_percent = gpu_mem
            except:
                pass

        return max(ram_percent, gpu_percent)

    def _adjust_params(self, latency, actual_batch_size):
        """
        ✅ IMPROVED: Điều chỉnh dựa trên actual batch size
        """
        q_len = len(self.queue)
        self._last_latency = latency
        self._last_run = time.time()

        memory_usage = self._get_memory_usage()

        # ✅ Log trước khi adjust
        # logger.debug(
            # f"[BatchManager] Before adjust: batch_size={self.batch_size}, "
            # f"actual={actual_batch_size}, queue={q_len}, "
            # f"latency={latency:.3f}s, memory={memory_usage:.2%}"
        # )

        # 1️⃣ Điều chỉnh batch_size
        if memory_usage > self.memory_threshold:
            # ⚠️ Memory cao: Giảm batch size
            new_batch = max(self.batch_size // 2, self.min_batch)
            if new_batch != self.batch_size:
                # logger.info(f"[BatchManager] Memory high ({memory_usage:.1%}), reducing batch: {self.batch_size} → {new_batch}")
                self.batch_size = new_batch
                
        elif q_len > self.batch_size * 2 and latency < self.target_latency:
            # ✅ Queue dài, latency OK: Tăng batch
            new_batch = min(self.batch_size * 2, self.max_batch)
            if new_batch != self.batch_size:
                # logger.info(f"[BatchManager] Queue long ({q_len}), increasing batch: {self.batch_size} → {new_batch}")
                self.batch_size = new_batch
                
        elif actual_batch_size < self.batch_size / 2 and q_len < self.min_batch:
            # ✅ Thực tế batch nhỏ + queue ngắn: Giảm batch
            new_batch = max(self.batch_size // 2, self.min_batch)
            if new_batch != self.batch_size:
                # logger.info(f"[BatchManager] Low traffic, reducing batch: {self.batch_size} → {new_batch}")
                self.batch_size = new_batch

        # 2️⃣ Điều chỉnh max_wait
        if latency > self.target_latency or memory_usage > self.memory_threshold:
            # ⚠️ Latency cao: Giảm wait time
            new_wait = max(self.min_wait, self.max_wait * 0.7)
            if abs(new_wait - self.max_wait) > 0.001:
                # logger.debug(f"[BatchManager] High latency, reducing wait: {self.max_wait:.3f}s → {new_wait:.3f}s")
                self.max_wait = new_wait
                
        elif q_len > self.batch_size:
            # ✅ Queue dài: Tăng wait time
            new_wait = min(0.2, self.max_wait * 1.2)
            if abs(new_wait - self.max_wait) > 0.001:
                # logger.debug(f"[BatchManager] Queue long, increasing wait: {self.max_wait:.3f}s → {new_wait:.3f}s")
                self.max_wait = new_wait

    def get_stats(self):
        """Lấy thống kê hiện tại"""
        avg_batch_size = self._total_items / self._total_batches if self._total_batches > 0 else 0
        
        return {
            "queue_length": len(self.queue),
            "batch_size": self.batch_size,
            "max_wait": self.max_wait,
            "last_latency": self._last_latency,
            "memory_usage": self._get_memory_usage(),
            "total_batches": self._total_batches,
            "total_items": self._total_items,
            "avg_batch_size": avg_batch_size,
        }