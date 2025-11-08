# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This code is referred from:
https://github.com/WenmuZhou/DBNet.pytorch/blob/master/post_processing/seg_detector_representer.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
# import paddle
from shapely.geometry import Polygon
import pyclipper
import torch
import numpy as np
from PIL import Image
from typing import (
    Tuple, List
    )
import io
import base64
from core.logger import logger
class DBPostProcess(object):
    """
    The post process for Differentiable Binarization (DB).
    """

    def __init__(
        self,
        thresh=0.3,
        box_thresh=0.7,
        max_candidates=1000,
        unclip_ratio=2.0,
        use_dilation=False,
        score_mode="fast",
        box_type="quad",
        **kwargs,
    ):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.score_mode = score_mode
        self.box_type = box_type
        assert score_mode in [
            "slow",
            "fast",
        ], "Score mode must be in [slow, fast] but got: {}".format(score_mode)

        self.dilation_kernel = None if not use_dilation else np.array([[1, 1], [1, 1]])

    def polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        """
        _bitmap: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
        """

        bitmap = _bitmap
        height, width = bitmap.shape

        boxes = []
        scores = []

        contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours[: self.max_candidates]:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue

            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            if points.shape[0] > 2:
                box = self.unclip(points, self.unclip_ratio)
                if len(box) > 1:
                    continue
            else:
                continue
            box = np.array(box).reshape(-1, 2)
            if len(box) == 0:
                continue

            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue

            box = np.array(box)
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height
            )
            boxes.append(box.tolist())
            scores.append(score)
        return boxes, scores

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        """
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        """

        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours(
            (bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            if self.score_mode == "fast":
                score = self.box_score_fast(pred, points.reshape(-1, 2))
            else:
                score = self.box_score_slow(pred, contour)
            if self.box_thresh > score:
                continue

            box = self.unclip(points, self.unclip_ratio)
            if len(box) > 1:
                continue
            box = np.array(box).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height
            )
            boxes.append(box.astype("int32"))
            scores.append(score)
        return np.array(boxes, dtype="int32"), scores

    def unclip(self, box, unclip_ratio):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(distance)
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        """
        box_score_fast: use bbox mean score as the mean score
        """
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]

    def box_score_slow(self, bitmap, contour):
        """
        box_score_slow: use polyon mean score as the mean score
        """
        h, w = bitmap.shape[:2]
        contour = contour.copy()
        contour = np.reshape(contour, (-1, 2))

        xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin

        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype("int32"), 1)
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]

    def __call__(self, outs_dict, shape_list):
        pred = outs_dict["maps"]
        if isinstance(pred, torch.Tensor):
            pred = pred.numpy()
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh

        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            if self.dilation_kernel is not None:
                mask = cv2.dilate(
                    np.array(segmentation[batch_index]).astype(np.uint8),
                    self.dilation_kernel,
                )
            else:
                mask = segmentation[batch_index]
            if self.box_type == "poly":
                boxes, scores = self.polygons_from_bitmap(
                    pred[batch_index], mask, src_w, src_h
                )
            elif self.box_type == "quad":
                boxes, scores = self.boxes_from_bitmap(
                    pred[batch_index], mask, src_w, src_h
                )
            else:
                raise ValueError("box_type can only be one of ['quad', 'poly']")

            boxes_batch.append({"points": boxes})
        return boxes_batch


class DistillationDBPostProcess(object):
    def __init__(
        self,
        model_name=["student"],
        key=None,
        thresh=0.3,
        box_thresh=0.6,
        max_candidates=1000,
        unclip_ratio=1.5,
        use_dilation=False,
        score_mode="fast",
        box_type="quad",
        **kwargs,
    ):
        self.model_name = model_name
        self.key = key
        self.post_process = DBPostProcess(
            thresh=thresh,
            box_thresh=box_thresh,
            max_candidates=max_candidates,
            unclip_ratio=unclip_ratio,
            use_dilation=use_dilation,
            score_mode=score_mode,
            box_type=box_type,
        )

    def __call__(self, predicts, shape_list):
        results = {}
        for k in self.model_name:
            results[k] = self.post_process(predicts[k], shape_list=shape_list)
        return results

def preprocess_image_config(image: Image.Image, resize_long: int = 960) -> Tuple[np.ndarray, List]:
    """
    Preprocessing chính xác theo config YAML
    """
    # 1. DecodeImage: Convert PIL (RGB) → BGR numpy array, channel_first=false (HWC)
    image_np = np.array(image)  # RGB
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # → BGR
    
    original_h, original_w = image_bgr.shape[:2]
    
    # 2. DetResizeForTest: resize_long=960 (resize theo cạnh dài nhất, giữ tỷ lệ)
    scale_ratio = resize_long / max(original_h, original_w)
    if scale_ratio != 1.0:
        new_h = int(original_h * scale_ratio)
        new_w = int(original_w * scale_ratio)
        image_bgr = cv2.resize(image_bgr, (new_w, new_h))
    
    # Padding để tránh lỗi model (chia hết cho 32)
    h, w = image_bgr.shape[:2]
    # pad_h = (32 - h % 32) % 32
    # pad_w = (32 - w % 32) % 32
    # if pad_h > 0 or pad_w > 0:
    #     image_bgr = cv2.copyMakeBorder(
    #         image_bgr, 0, pad_h, 0, pad_w, 
    #         cv2.BORDER_CONSTANT, value=(0, 0, 0)  
    #     )
    dh, dw = resize_long - h, resize_long - w
    dh /= 2
    dw /= 2
    image_bgr = cv2.copyMakeBorder(image_bgr, int(dh), int(dh + 0.5), int(dw), int(dw + 0.5),
                       cv2.BORDER_CONSTANT, value=(0, 0, 0))
    new_ratio_h = image_bgr.shape[0] / original_h
    new_ratio_w = image_bgr.shape[1] / original_w
    # 3. NormalizeImage: order=hwc, scale=1./255, mean & std
    image_float = image_bgr.astype(np.float32)
    image_float = image_float / 255.0  # scale = 1./255
    
    # Normalize theo HWC order
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # RGB order trong config
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    # Nhưng ảnh là BGR, nên reverse mean/std
    mean_bgr = mean[::-1]  # [0.406, 0.456, 0.485]
    std_bgr = std[::-1]    # [0.225, 0.224, 0.229]
    
    image_float = (image_float - mean_bgr) / std_bgr
    
    # 4. ToCHWImage: HWC → CHW
    image_chw = np.transpose(image_float, (2, 0, 1))  # HWC → CHW
    
    # 5. Add batch dimension
    image_batch = np.expand_dims(image_chw, axis=0)  # CHW → BCHW
    
    # Shape info cho postprocess
    shape_info = [original_h, original_w, new_ratio_h, new_ratio_w]

    return image_batch, shape_info, (int(dh + 0.5)/new_ratio_h, int(dw + 0.5)/new_ratio_w)

def base64_to_pil(base64_str: str) -> Image.Image:
    # Remove the "data:image/xxx;base64," prefix if present
    if base64_str.startswith("data:image"):
        base64_str = base64_str.split(",")[1]
    
    # Decode the base64 string
    image_data = base64.b64decode(base64_str)
    
    # Open the image using PIL
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    return image

def crop_image(image, bbox, dwh = (0, 0), scale = (1.0, 1.0), pad=0, use_scale=False):
    """Crop ảnh theo bbox (x_min, y_min, x_max, y_max) với padding."""
    try:
        # ✅ Validate input
        x_min, y_min, x_max, y_max = bbox
        h, w = image.shape[:2]
        
        # ✅ Apply offset and scale
        dh, dw = dwh
        if use_scale:
            x_min = int((x_min - dw) / scale[-2])
            y_min = int((y_min - dh) / scale[-1])
            x_max = int((x_max + dw) / scale[-2])
            y_max = int((y_max + dh) / scale[-1])
        else:
            x_min = int(x_min - dw)
            y_min = int(y_min - dh)
            x_max = int(x_max + dw)
            y_max = int(y_max + dh)

        # ✅ Clip to image bounds
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(w, x_max + pad)
        y_max = min(h, y_max + pad)
        
        # ✅ Validate final bbox
        if x_min >= x_max or y_min >= y_max:
            return None
        cropped = image[y_min:y_max, x_min:x_max]
        
        if cropped.size == 0:
            return None
        return cropped
        
    except Exception as e:
        logger.error(f"❌ crop_image: Exception - {e}")
        return None