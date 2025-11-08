import onnxruntime as ort
from functools import lru_cache
import pathlib
from typing import List
from ultralytics import YOLO


def init_config():
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Tạo cache cho TensorRT (nếu dùng TRT EP)
    trt_cache = pathlib.Path("./trt_cache")
    trt_cache.mkdir(exist_ok=True)

    TRT_OPTS = {
        "trt_fp16_enable": True,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": "trt_cache",
        "trt_max_workspace_size": 1 << 28,
        "trt_force_sequential_engine_build": True,
        "trt_int8_enable": False,
        "trt_timing_cache_enable": True,
        "trt_engine_decryption_enable": False,
        "trt_builder_optimization_level": 5,
        "trt_dla_enable": False,
        "trt_dla_core": 0,
        "trt_enable_fallback": True,     # 🔥 thêm dòng này
    }

    CUDA_OPTS = {
        "device_id": 0,
        "cudnn_conv_algo_search": "HEURISTIC",
        "enable_cuda_graph": False,
    }

    # Ưu tiên: TensorRT -> CUDA -> CPU, nhưng sẽ lọc theo khả dụng
    PREFERRED_PROVIDERS = [
        ("TensorrtExecutionProvider", TRT_OPTS),
        ("CUDAExecutionProvider", CUDA_OPTS),
        "CPUExecutionProvider",
    ]

    return so, TRT_OPTS, CUDA_OPTS, PREFERRED_PROVIDERS


def resolve_providers(preferred) -> List:
    """Chỉ dùng các provider hiện có để tránh lỗi load .so."""
    available = set(ort.get_available_providers())
    resolved = []
    for p in preferred:
        name = p if isinstance(p, str) else p[0]
        if name in available:
            resolved.append(p)
    # luôn đảm bảo có CPU fallback
    if "CPUExecutionProvider" not in [
        p if isinstance(p, str) else p[0] for p in resolved
    ]:
        resolved.append("CPUExecutionProvider")
    return resolved


@lru_cache(maxsize=None)
def load_model(model_path: str, type="onnx"):
    """Load ONNX model (cache 1 lần duy nhất)."""
    if type == "pt":
        return YOLO(model_path, task="detect")
    so, TRT_OPTS, CUDA_OPTS, PREFERRED_PROVIDERS = init_config()
    providers = resolve_providers(PREFERRED_PROVIDERS)
    path = str(pathlib.Path(model_path).resolve())

    # ✅ Log providers
    print(f"[MODEL MANAGER] Loading: {model_path}")
    print(f"[MODEL MANAGER] Available providers: {ort.get_available_providers()}")
    print(
        f"[MODEL MANAGER] Using providers: {[p if isinstance(p, str) else p[0] for p in providers]}"
    )

    session = ort.InferenceSession(path, sess_options=so, providers=providers)

    # ✅ Log actual provider used
    actual_provider = session.get_providers()[0]
    print(f"[MODEL MANAGER] ✅ Model loaded with provider: {actual_provider}")

    if actual_provider == "CPUExecutionProvider":
        print(f"[MODEL MANAGER] ⚠️  WARNING: Running on CPU! GPU not used!")
    elif actual_provider == "TensorrtExecutionProvider":
        print(f"[MODEL MANAGER] 🚀 OPTIMAL: Running on GPU with TensorRT (FP16)")
    elif actual_provider == "CUDAExecutionProvider":
        print(f"[MODEL MANAGER] ✅ Running on GPU with CUDA")

    return session
