from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from services.task_service import TaskService

router = APIRouter(prefix="/api", tags=["Upload"])


class ImageBase64Request(BaseModel):
    """Request body với image base64"""

    image: str  # Base64 encoded image


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    task_id = TaskService.create_task(image_bytes)
    return {"task_id": task_id, "status": "queued"}


@router.post("/upload/image")
async def upload_image_base64(request: ImageBase64Request):
    """
    ✅ Upload image từ base64 string (for testing & API clients)

    Body:
        {"image": "base64_string_here"}

    Returns:
        {"task_id": "...", "status": "queued"}
    """
    import base64

    try:
        # Decode base64
        image_bytes = base64.b64decode(request.image)

        # Create task
        task_id = TaskService.create_task(image_bytes)

        return {"task_id": task_id, "status": "queued"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


@router.get("/result/{task_id}")
async def get_result(task_id: str):
    result = TaskService.fetch_result(task_id)
    if result:
        return {"task_id": task_id, "result": result}
    return {"task_id": task_id, "status": "processing"}
