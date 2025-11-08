import redis
import json
from core.config import settings

# ✅ Sử dụng connection pool để tối ưu performance
redis_pool = redis.ConnectionPool(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB,
    max_connections=50,
    decode_responses=True,
)

redis_client = redis.Redis(connection_pool=redis_pool)


def push_task(task_data: dict):
    redis_client.lpush("task_queue", json.dumps(task_data))


def pop_task(timeout=0):
    data = redis_client.brpop("task_queue", timeout=timeout)
    if data:
        _, task_json = data
        return json.loads(task_json)
    return None


def save_result(task_id: str, result: dict):
    redis_client.set(f"result:{task_id}", json.dumps(result))


def get_result(task_id: str):
    result = redis_client.get(f"result:{task_id}")
    return json.loads(result) if result else None
