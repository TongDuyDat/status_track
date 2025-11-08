import asyncio
import cv2
import numpy as np
import torch
import os

from pipelines.default import OCRResult, PipelineResult, Result_, TruckResult
from .track_detect_pipeline import truck_dectect
from .text_dectect_pipeline import text_detect
from .text_recognition_pipeline import ocr_inference
from .adaptive_batch_manager import AdaptiveBatchManager
from .model_manager import load_model
from .utils import crop_image
from core.config import settings
import uuid
from core.logger import logger

# Debug mode từ environment variable
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# ===== Load models =====
text_det_session = load_model("text_detection/inference.onnx")
ocr_session = load_model(
    "text_recognition/pretrained=parseq-patch16-224_prepost_process_v2.onnx"
)
yolo_model = load_model("track_model/best.pt", type="pt")

# ===== Create adaptive batch managers với memory management =====
truck_batcher = AdaptiveBatchManager(
    lambda imgs: truck_dectect(yolo_model, imgs),
    min_batch=1,
    max_batch=settings.TRUCK_MAX_BATCH,
    target_latency=0.1,
    max_queue_size=settings.MAX_QUEUE_SIZE,
    memory_threshold=settings.GPU_THRESHOLD,
)

text_batcher = AdaptiveBatchManager(
    lambda imgs: [text_detect(text_det_session, im) for im in imgs],
    min_batch=1,
    max_batch=settings.TEXT_MAX_BATCH,
    target_latency=0.08,
    max_queue_size=settings.MAX_QUEUE_SIZE,
    memory_threshold=settings.GPU_THRESHOLD,
)

ocr_batcher = AdaptiveBatchManager(
    lambda imgs: ocr_inference(ocr_session, imgs),
    min_batch=4,
    max_batch=settings.OCR_MAX_BATCH,
    target_latency=0.05,
    max_queue_size=settings.MAX_QUEUE_SIZE,
    memory_threshold=settings.GPU_THRESHOLD,
)


# ===== Model warm-up =====
def warm_up_models():
    """Pre-run inference để build TensorRT engines và khởi tạo CUDA"""
    print("🔥 Warming up models...")
    dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Warm-up truck detection
    truck_dectect(yolo_model, [dummy_img])

    # Warm-up text detection
    text_detect(text_det_session, dummy_img)

    # Warm-up OCR
    dummy_crop = np.random.randint(0, 255, (32, 224, 3), dtype=np.uint8)
    ocr_inference(ocr_session, [dummy_crop])

    print("✅ Models warmed up!")


warm_up_models()


async def pipeline_async(images):
    """Full async pipeline: Truck detect → Text detect → OCR"""
    results_json = []

    # 1️⃣ Truck detection
    truck_futs = [truck_batcher.infer(im) for im in images]
    truck_outputs = await asyncio.gather(*truck_futs)
    truck_results = [TruckResult(det).process() for det in truck_outputs]
    
    cropped_plates = []
    valid_pairs = []  # giữ (ảnh, truck_result)
    for im, res in zip(images, truck_results):
        if len(res.plate_bbox) > 0:
            # logger.info(f"[Pipeline] Detected plate bbox: {res.plate_bbox}")
            plate = crop_image(im, res.plate_bbox)
            if plate is None:
                continue
            cropped_plates.append(plate)
            valid_pairs.append((im, res))
    # ✅ Early return nếu không có plates
    # logger.info(f"[Pipeline] Detected {len(cropped_plates)} plates in {len(images)} images")
    if not cropped_plates:
        return []
                
    # 2️⃣ Text detection (multi-bbox)
    # logger.info("[Pipeline] Running text detection...")
    text_outputs = await asyncio.gather(
        *[text_batcher.infer(im) for im in cropped_plates]
    )
    # Mỗi text_output có thể chứa nhiều bbox (list các vùng chữ)
    # logger.info("[Pipeline] Processing text regions...")
    cropped_text_groups = []  # [[img_crop_1, img_crop_2, ...], [img_crop_1, ...], ...]
    for plate, det in zip(cropped_plates, text_outputs):
        bboxes = det
        # Sắp xếp từ trái qua phải (theo x1)
        # bboxes.sort(key=lambda b: b[0])
        crops = [crop_image(plate, box, (dh, dw), scale, use_scale=False) for box, (dh, dw), scale in bboxes]
        cropped_text_groups.append(crops)
    # 3️⃣ OCR recognition (OPTIMIZED - Flatten batching)
    ocr_results = []
    # ✅ Flatten tất cả crops thành 1 list để batch 1 lần duy nhất
    all_crops = []
    crop_counts = []  # Track số crops của mỗi plate
    for crops in cropped_text_groups:
        if not crops:
            crop_counts.append(0)
        else:
            all_crops.extend(crops)
            crop_counts.append(len(crops))

    # ✅ Batch OCR 1 lần cho tất cả text regions
    if all_crops:
        all_ocr_outputs = await asyncio.gather(
            *[ocr_batcher.infer(im) for im in all_crops]
        )

        # ✅ Group lại theo từng plate
        idx = 0
        for i, count in enumerate(crop_counts):
            if count == 0:
                ocr_results.append(OCRResult({"text": "", "conf": 0.0}).process())
            else:
                # Lấy OCR results của plate hiện tại
                plate_ocr_outputs = all_ocr_outputs[idx : idx + count]
                ocr_items = [OCRResult(det).process() for det in plate_ocr_outputs]

                # Debug: Lưu ảnh chỉ khi DEBUG_MODE=true
                if DEBUG_MODE:
                    for ocr_item, groups, plate in zip(ocr_items, cropped_text_groups, cropped_plates):
                        name = uuid.uuid4().hex[0:5]
                        cv2.imwrite(f"images_test/plate_{name}.jpg", plate)
                        for i in groups:
                            cv2.imwrite(f"images_test/ocr_{ocr_item.text}_{name}.jpg", i)

                # Ghép text theo thứ tự
                full_text = "".join([o.text for o in ocr_items])
                mean_conf = (
                    sum([o.confidence for o in ocr_items]) / len(ocr_items)
                    if ocr_items
                    else 0.0
                )
                ocr_results.append(
                    OCRResult({"text": full_text, "conf": mean_conf}).process()
                )

                idx += count
    else:
        # Không có text regions nào được detect
        ocr_results = [
            OCRResult({"text": "", "conf": 0.0}).process() for _ in cropped_text_groups
        ]

    # 4️⃣ Gộp kết quả
    for (im, truck_res), ocr_res in zip(valid_pairs, ocr_results):
        final_res = PipelineResult(truck_res, ocr_res)
        results_json.append(final_res.to_dict())

    return results_json
