import onnx

# Tự định nghĩa ánh xạ mã dtype → tên kiểu NumPy
ONNX_DTYPE_MAP = {
    1: "float32",
    2: "uint8",
    3: "int8",
    4: "uint16",
    5: "int16",
    6: "int32",
    7: "int64",
    9: "bool",
    10: "float16",
    11: "double",
    12: "uint32",
    13: "uint64",
    14: "complex64",
    15: "complex128",
    16: "bfloat16",
}

model_path = "text_recognition/pretrained=parseq-patch16-224_prepost_process_v2.onnx"
model = onnx.load(model_path)
onnx.checker.check_model(model)

print("=== INPUTS ===")
for i, input_tensor in enumerate(model.graph.input):
    shape = [
        d.dim_value if (d.dim_value > 0) else "dynamic"
        for d in input_tensor.type.tensor_type.shape.dim
    ]
    dtype_code = input_tensor.type.tensor_type.elem_type
    dtype = ONNX_DTYPE_MAP.get(dtype_code, f"unknown({dtype_code})")
    print(f"{i}. {input_tensor.name} | dtype: {dtype} | shape: {shape}")

print("\n=== OUTPUTS ===")
for i, output_tensor in enumerate(model.graph.output):
    shape = [
        d.dim_value if (d.dim_value > 0) else "dynamic"
        for d in output_tensor.type.tensor_type.shape.dim
    ]
    dtype_code = output_tensor.type.tensor_type.elem_type
    dtype = ONNX_DTYPE_MAP.get(dtype_code, f"unknown({dtype_code})")
    print(f"{i}. {output_tensor.name} | dtype: {dtype} | shape: {shape}")
