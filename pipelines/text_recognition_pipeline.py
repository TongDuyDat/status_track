import pathlib
import time
from typing import List
import sys
# sys.path.append(str(pathlib.Path(__file__).parent.parent))
from pipelines.model_manager import load_model
from .default import CHAR
import onnxruntime as ort
import cv2
import numpy as np
import torch
from PIL import Image
from functools import singledispatch


# ===== Helper function xử lý 1 ảnh =====
def process_single_image(im: np.ndarray) -> np.ndarray:
    """Resize, normalize và chuyển ảnh sang [C,H,W] (chuẩn hóa OCR)."""
    target_h = 32  # ❌ XÓA target_w = 224

    # RGB đảm bảo
    if im.ndim == 2:
        im = np.stack([im] * 3, axis=-1)
    elif im.shape[2] == 4:  # RGBA
        im = im[:, :, :3]

    # ✅ Resize theo tỉ lệ, KHÔNG giới hạn width
    h, w, _ = im.shape
    scale = target_h / h
    new_w = max(1, int(w * scale))  # ✅ Bỏ min(new_w, 224)
    im = cv2.resize(im, (new_w, target_h))

    # ❌ BỎ PHẦN PAD (mô hình chấp nhận width động)
    
    # [H, W, C] → [1, C, H, W]
    im = im.astype("float32") / 255.0
    im = np.expand_dims(im.transpose(2, 0, 1), axis=0)
    return im

# ===== Định nghĩa hàm pre_processing với singledispatch =====
@singledispatch
def pre_processing(image):
    """
    Hàm xử lý ảnh trước khi inference OCR.
    Hỗ trợ nhiều kiểu input: str (path), PIL.Image, np.ndarray, list.
    """
    raise NotImplementedError(f"Unsupported image type: {type(image)}")


@pre_processing.register(str)
def _(image_path: str):
    """Xử lý từ đường dẫn file."""
    im = np.array(Image.open(image_path).convert("RGB"))
    return process_single_image(im)


@pre_processing.register(Image.Image)
def _(image: Image.Image):
    """Xử lý từ PIL Image."""
    im = np.array(image.convert("RGB"))
    return process_single_image(im)


@pre_processing.register(np.ndarray)
def _(image: np.ndarray):
    """Xử lý từ numpy array."""
    return process_single_image(image)

@pre_processing.register(list)
def _(image: list):
    if not image:
        return np.empty((0, 3, 32, 0), dtype=np.float32)  # ✅ Width = 0 khi empty

    processed = []
    target_h = 32

    for im in image:
        # Convert to numpy if needed
        if isinstance(im, str):
            im = np.array(Image.open(im).convert("RGB"))
        elif isinstance(im, Image.Image):
            im = np.array(im.convert("RGB"))
        elif not isinstance(im, np.ndarray):
            continue

        # Ensure RGB
        if im.ndim == 2:
            im = np.stack([im] * 3, axis=-1)
        elif im.shape[2] == 4:
            im = im[:, :, :3]

        # ✅ Resize theo tỉ lệ, KHÔNG giới hạn width
        h, w = im.shape[:2]
        scale = target_h / h
        new_w = max(1, int(w * scale))
        im = cv2.resize(im, (new_w, target_h), interpolation=cv2.INTER_LINEAR)

        # ❌ BỎ PHẦN PAD
        processed.append(im)

    if not processed:
        return np.empty((0, 3, 32, 0), dtype=np.float32)

    # ✅ QUAN TRỌNG: Tìm max width trong batch
    max_w = max(im.shape[1] for im in processed)
    
    # Pad tất cả ảnh về cùng width = max_w
    padded = []
    for im in processed:
        pad_w = max_w - im.shape[1]
        if pad_w > 0:
            im = np.pad(im, ((0, 0), (0, pad_w), (0, 0)), constant_values=255)
        padded.append(im)

    # Stack and normalize
    batch = np.stack(padded, axis=0)  # (B, H, max_W, C)
    batch = batch.astype(np.float32) / 255.0
    batch = batch.transpose(0, 3, 1, 2)  # (B, C, H, max_W)

    return batch


def post_processing(results):
    token_ids = torch.from_numpy(results[1])
    probs = torch.from_numpy(results[0])
    bs, seq = token_ids.shape
    decode_text = []
    mask = (token_ids > 0) & (token_ids < len(CHAR))

    eos_mask = token_ids == 0
    eos_index = eos_mask.float().argmax(dim=1)
    eos_index[eos_mask.sum(dim=1) == 0] = seq

    for b in range(bs):
        vaild = token_ids[b, : eos_index[b]][mask[b, : eos_index[b]]]
        decode_text.append("".join([CHAR[i] for i in vaild.tolist()]))
    return decode_text


def ocr_inference(session: ort.InferenceSession, images) -> str:
    """Chạy OCR trên patch ảnh (BGR hoặc RGB) và trả về danh sách dict {'text', 'conf'}."""
    try:
        # 1️⃣ Tiền xử lý
        image_np = pre_processing(images)

        # 2️⃣ Inference
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: image_np})

        # 3️⃣ Giải mã text (token → chuỗi)
        text_results = post_processing(outputs)

        # 4️⃣ Chuẩn hóa output về dạng dict
        results = []
        for txt in text_results:
            if isinstance(txt, str):
                results.append({"text": txt, "conf": 1.0})
            elif isinstance(txt, dict):
                # Đề phòng post_processing đã trả dict có conf
                text_val = txt.get("text", "")
                conf_val = float(txt.get("conf", 1.0))
                results.append({"text": text_val, "conf": conf_val})
            else:
                results.append({"text": str(txt), "conf": 0.0})

        return results

    except Exception as e:
        print(f"[OCR] Inference error: {e}")
        # nếu images là list → trả list rỗng cùng kích thước
        if isinstance(images, list):
            return [{"text": "", "conf": 0.0} for _ in images]
        else:
            return [{"text": "", "conf": 0.0}]
