import aioredis
import json
from core.config import settings

# ✅ Async Redis client
redis_async = None
from loguru import logger
async def get_async_redis():
    global redis_async
    if redis_async is None:
        redis_async = await aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )
    return redis_async

async def pop_task_async(timeout: float = 1.0):
    """Non-blocking async Redis pop"""
    redis = await get_async_redis()
    try:
        # ✅ blpop with timeout (non-blocking for event loop)
        result = await redis.blpop("task_queue", timeout=timeout)
        if result:
            queue_name, task_json = result
            return json.loads(task_json)
    except Exception as e:
        logger.error(f"Redis pop error: {e}")
    return None

async def save_result_async(task_id: str, result: dict):
    """Save result to Redis (non-blocking async)"""
    redis = await get_async_redis()
    try:
        await redis.setex(
            f"result:{task_id}",
            settings.RESULT_EXPIRE,  # ✅ Now works
            json.dumps(result)
        )
        logger.debug(f"✅ Saved result for task {task_id}")
    except Exception as e:
        logger.error(f"Redis save error for task {task_id}: {e}")