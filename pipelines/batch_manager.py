# pipelines/batch_manager.py
import asyncio
import numpy as np

class BatchManager:
    """Batch inference manager (async, safe init)."""
    def __init__(self, infer_fn, batch_size=8, max_wait=0.05):
        self.infer_fn = infer_fn
        self.batch_size = batch_size
        self.max_wait = max_wait
        self.queue = []
        self.lock = asyncio.Lock()
        self._started = False  # ⚡ cờ khởi động loop

    async def _loop(self):
        while True:
            await asyncio.sleep(self.max_wait)
            async with self.lock:
                if not self.queue:
                    continue
                batch = self.queue[:self.batch_size]
                self.queue = self.queue[self.batch_size:]
            inputs, futs = zip(*batch)
            results = self.infer_fn(list(inputs))
            for fut, res in zip(futs, results):
                fut.set_result(res)

    async def infer(self, data):
        # ⚡ đảm bảo chỉ khởi động 1 lần trong event loop thực
        if not self._started:
            asyncio.create_task(self._loop())
            self._started = True
        fut = asyncio.get_event_loop().create_future()
        async with self.lock:
            self.queue.append((data, fut))
        return await fut
