from typing import List, Union
import cv2
import onnxruntime as ort    
from PIL import Image
import torch
import numpy as np
from .utils import preprocess_image_config, crop_image
from .default import postprocess
from core.logger import logger

# Sắp xếp bbox theo khoảng cách từ top-left (x_min, y_min) tới gốc (0,0)
def sort_bboxes_by_rows_tensor(bboxes: torch.Tensor, y_threshold: float = 20.0):
    """
    Sắp xếp bbox theo hàng (row), từ trên xuống dưới, trái sang phải.
    """
    # 1) Lấy góc top-left của mỗi bbox
    x_min = bboxes[:, :, 0].min(dim=1).values  # (N,)
    y_min = bboxes[:, :, 1].min(dim=1).values  # (N,)
    # 2) Sort sơ bộ theo y (trên → dưới)
    sorted_y, idx_y = torch.sort(y_min)
    bboxes = bboxes[idx_y]
    x_min = x_min[idx_y]
    y_min = sorted_y
    # 3) Gom thành các hàng (rows) dựa trên y_threshold
    # Nếu 2 bbox liên tiếp có chênh lệch y > threshold → hàng mới
    diff_y = torch.abs(y_min[1:] - y_min[:-1]) > y_threshold
    row_id = torch.zeros_like(y_min, dtype=torch.long)
    row_id[1:] = torch.cumsum(diff_y, dim=0)  # Tăng dần khi gặp hàng mới
    # 4) Sort cuối cùng: ưu tiên row_id (hàng), sau đó x_min (trái→phải)
    row_weight = 1e6  # Đảm bảo row_id quan trọng hơn x_min
    scores = row_id * row_weight + x_min
    _, final_idx = torch.sort(scores)
    return bboxes[final_idx]

def model_inference(ort_session: ort.InferenceSession, input_tensor: torch.Tensor):
    input_name = ort_session.get_inputs()[0].name
    return ort_session.run(None, {input_name: input_tensor})


def text_detect_single(det_session: ort.InferenceSession, image, mode="xyxy"):
    """
    Detect text trong 1 ảnh (single image)
    
    Args:
        det_session: ONNX session
        image: str (path), PIL.Image, hoặc np.ndarray
        mode: "xyxy" hoặc "point"
    
    Returns:
        List[Tuple]: [(bbox, (dh, dw), scale), ...]
    """
    try:
        # Load ảnh & tiền xử lý
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            im_vis = np.array(image)
        elif isinstance(image, np.ndarray):
            im_vis = image
        else:
            logger.error(f"Invalid image type: {type(image)}")
            return []
        
        # ✅ Validate image
        if im_vis is None or im_vis.size == 0:
            logger.error("text_detect_single: Input image is empty")
            return []
        
        logger.debug(f"text_detect_single: Processing image shape={im_vis.shape}")
        
        # Preprocess
        inp, scale, (dh, dw) = preprocess_image_config(image.copy(), resize_long=640)
        
        # Inference
        outputs = model_inference(det_session, inp)
        
        # Postprocess
        pred = {"maps": outputs[0]}
        shape_list = [scale]
        outs = postprocess(pred, shape_list)
        
        bboxes_xyxy = []
        for out in outs:
            bboxes = out.get('points', [])
            bboxes = sort_bboxes_by_rows_tensor(torch.tensor(bboxes)) if len(bboxes) > 0 else bboxes
            
            if mode == "point":
                return bboxes
            
            if out and len(bboxes) > 0:
                for pts in bboxes:
                    pts = pts.astype(np.int32).reshape(-1, 2)
                    x_min, y_min = np.clip(pts.min(axis=0), a_min=0, a_max=None)
                    x_max = min(im_vis.shape[1], pts[:, 0].max())
                    y_max = min(im_vis.shape[0], pts[:, 1].max())
                    
                    if x_max > x_min and y_max > y_min:
                        bboxes_xyxy.append(((x_min, y_min, x_max, y_max), (dh, dw), scale))
        
        logger.debug(f"text_detect_single: Found {len(bboxes_xyxy)} text regions")
        return bboxes_xyxy
    
    except Exception as e:
        logger.error(f"text_detect_single: Exception - {e}")
        return []


def text_detect_batch(det_session: ort.InferenceSession, images: List[np.ndarray], mode="xyxy"):
    """
    ✅ Batch text detection cho nhiều ảnh
    
    Args:
        det_session: ONNX session
        images: List[np.ndarray] - danh sách ảnh
        mode: "xyxy" hoặc "point"
    
    Returns:
        List[List[Tuple]]: Danh sách kết quả cho từng ảnh
    """
    if not images:
        logger.warning("text_detect_batch: Empty image list")
        return []
    
    logger.debug(f"text_detect_batch: Processing {len(images)} images")
    
    try:
        # ✅ Preprocess tất cả ảnh
        preprocessed = []
        metadata = []  # Lưu (scale, dh, dw, original_shape)
        
        for idx, img in enumerate(images):
            if img is None or img.size == 0:
                logger.warning(f"text_detect_batch: Image {idx} is empty, skipping")
                preprocessed.append(None)
                metadata.append(None)
                continue
            
            inp, scale, (dh, dw) = preprocess_image_config(img.copy(), resize_long=640)
            preprocessed.append(inp)
            metadata.append((scale, dh, dw, img.shape))
        
        # ✅ Filter valid inputs
        valid_indices = [i for i, inp in enumerate(preprocessed) if inp is not None]
        if not valid_indices:
            logger.error("text_detect_batch: No valid images to process")
            return [[] for _ in images]
        
        # ✅ Stack thành batch (tất cả input đã có cùng kích thước sau preprocess)
        valid_inputs = [preprocessed[i] for i in valid_indices]
        valid_metadata = [metadata[i] for i in valid_indices]
        
        # Concatenate along batch dimension
        batch_input = np.concatenate(valid_inputs, axis=0)  # (B, C, H, W)
        logger.debug(f"text_detect_batch: Batch input shape={batch_input.shape}")
        
        # ✅ Batch inference
        input_name = det_session.get_inputs()[0].name
        batch_outputs = det_session.run(None, {input_name: batch_input})
        
        # ✅ Postprocess từng ảnh trong batch
        all_results = [[] for _ in images]  # Khởi tạo kết quả cho tất cả ảnh
        
        for batch_idx, original_idx in enumerate(valid_indices):
            try:
                # Lấy output cho ảnh thứ batch_idx
                single_output = batch_outputs[0][batch_idx:batch_idx+1]  # (1, H, W)
                
                scale, dh, dw, orig_shape = valid_metadata[batch_idx]
                
                # Postprocess
                pred = {"maps": single_output}
                shape_list = [scale]
                outs = postprocess(pred, shape_list)
                
                bboxes_xyxy = []
                for out in outs:
                    bboxes = out.get('points', [])
                    bboxes = sort_bboxes(np.array(bboxes)) if len(bboxes) > 0 else bboxes
                    
                    if mode == "point":
                        all_results[original_idx] = bboxes
                        continue
                    
                    if out and len(bboxes) > 0:
                        im_h, im_w = orig_shape[:2]
                        for pts in out["points"]:
                            pts = pts.astype(np.int32).reshape(-1, 2)
                            x_min, y_min = np.clip(pts.min(axis=0), a_min=0, a_max=None)
                            x_max = min(im_w, pts[:, 0].max())
                            y_max = min(im_h, pts[:, 1].max())
                            
                            if x_max > x_min and y_max > y_min:
                                bboxes_xyxy.append(((x_min, y_min, x_max, y_max), (dh, dw), scale))
                
                all_results[original_idx] = bboxes_xyxy
                logger.debug(f"text_detect_batch: Image {original_idx} -> {len(bboxes_xyxy)} regions")
                
            except Exception as e:
                logger.error(f"text_detect_batch: Error processing image {original_idx} - {e}")
                all_results[original_idx] = []
        
        logger.info(f"text_detect_batch: Processed {len(valid_indices)}/{len(images)} images")
        return all_results
    
    except Exception as e:
        logger.error(f"text_detect_batch: Batch inference failed - {e}")
        # Fallback: Process từng ảnh riêng lẻ
        logger.warning("text_detect_batch: Falling back to single image processing")
        return [text_detect_single(det_session, img, mode) for img in images]


def text_detect(det_session: ort.InferenceSession, images: Union[np.ndarray, List[np.ndarray]], mode="xyxy"):
    """
    ✅ Wrapper function: Tự động detect single hoặc batch
    
    Args:
        det_session: ONNX session
        images: np.ndarray (single) hoặc List[np.ndarray] (batch)
        mode: "xyxy" hoặc "point"
    
    Returns:
        Single: List[Tuple]
        Batch: List[List[Tuple]]
    """
    # ✅ Detect input type
    if isinstance(images, list):
        # Batch processing
        return text_detect_batch(det_session, images, mode)
    else:
        # Single image
        return text_detect_single(det_session, images, mode)