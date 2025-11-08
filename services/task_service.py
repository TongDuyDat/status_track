import uuid
from utils.redis_utils import push_task, get_result

class TaskService:
    @staticmethod
    def create_task(image_bytes: bytes) -> str:
        task_id = str(uuid.uuid4())
        push_task({
            "task_id": task_id,
            "image_bytes": image_bytes.hex()
        })
        return task_id

    @staticmethod
    def fetch_result(task_id: str):
        return get_result(task_id)
