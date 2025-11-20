import numpy as np
import torch
from .utils import DBPostProcess
from abc import ABC
from typing import Dict, Any
CHAR = (
    '[E]', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
    'x', 'y', 'z',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
    'X', 'Y', 'Z',
    '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[',
    '\\', ']', '^', '_', '`', '{', '|', '}', '~', '[B]', '[P]'
)

postprocess = DBPostProcess(thresh=0.4,
                             box_thresh=0.6,
                             max_candidates=1000,
                             unclip_ratio=1.5)

CLASS = {
    0: "Plate",
    1: "Thùng xe có hàng",
    2: "Thùng xe không có chứa hàng",
}

# pipelines/result_objects.py
from abc import ABC, abstractmethod
from typing import Any, List, Dict

class Result_(ABC):
    def __init__(self, data: Any = None):
        super().__init__()
        self.data = data
        self.bbox: List[float] = []
        self.status: str = ""
        self.meta: Dict[str, Any] = {}

    @abstractmethod
    def process(self):
        pass


class TruckResult(Result_):
    def __init__(self, data: Any = None):
        super().__init__(data)
        self.confidence: float = 0.0

    def _convert_to_python(self, value):
        """Chuyển tensor/numpy về Python native type."""
        if isinstance(value, (torch.Tensor, np.ndarray)):
            # Nếu là array/tensor → list
            if value.ndim > 0:
                return value.tolist()
            # Nếu là scalar tensor → float/int
            return value.item()
        return value

    def process(self):
        det = self.data or {}
        
        # ✅ Chuyển đổi bbox về list
        self.plate_bbox = self._convert_to_python(det.get("plate_bbox", []))
        self.status_bbox = self._convert_to_python(det.get("status_bbox", []))
        
        # ✅ Chuyển đổi conf về float
        self.plate_conf = float(self._convert_to_python(det.get("plate_conf", 0.0)))
        self.status_conf = float(self._convert_to_python(det.get("status_conf", 0.0)))
        self.status_idx = int(self._convert_to_python(det.get("status_idx", 0)))
        
        self.status = CLASS.get(self.status_idx, "")
        return self


class OCRResult(Result_):
    def __init__(self, data: Any = None):
        super().__init__(data)
        self.text: str = ""
        self.confidence: float = 0.0

    def _convert_to_python(self, value):
        """Chuyển tensor/numpy về Python native type."""
        if isinstance(value, (torch.Tensor, np.ndarray)):
            if value.ndim > 0:
                return value.tolist()
            return value.item()
        return value

    def process(self):
        ocr = self.data or {}
        self.text = str(ocr.get("text", ""))
        
        # ✅ Chuyển conf về float
        self.confidence = float(self._convert_to_python(ocr.get("conf", 0.0)))
        
        self.status = "recognized"
        self.meta["ocr_conf"] = self.confidence
        return self


class PipelineResult:
    """Gộp kết quả cuối cùng (truck + ocr)"""
    def __init__(self, truck: TruckResult, ocr: OCRResult):
        self.plate_number = ocr.text
        self.plate_bbox = truck.plate_bbox
        self.truck_conf = float(truck.status_conf)  # ✅ Fix: Lấy từ status_conf
        self.ocr_conf = float(ocr.confidence)
        self.truck_status = truck.status
        self.truck_bbox = truck.status_bbox

    def to_dict(self):
        """Đảm bảo tất cả giá trị đều JSON serializable."""
        return {
            "plate_number": str(self.plate_number),
            "plate_bbox": self.plate_bbox if isinstance(self.plate_bbox, list) else [],
            "truck_conf": float(self.truck_conf),
            "ocr_conf": float(self.ocr_conf),
            "truck_status": str(self.truck_status),
            "truck_bbox": self.truck_bbox if isinstance(self.truck_bbox, list) else [],
        }
