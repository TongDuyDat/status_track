import pathlib
import torch
from ultralytics import YOLO


def post_processing(preds):
    results = []
    for pred in preds:
        data = pred.boxes.data
        if data.numel() == 0:
            results.append(
                {"plate": torch.empty((0, 6)), "status": torch.empty((0, 6))}
            )
            continue
        # Tạo mask theo class ID
        plate_mask = data[:, -1] == 0
        status_mask = ~plate_mask

        # mask theo class
        plate_mask = data[:, -1] == 0
        status_mask = ~plate_mask

        plate_det = data[plate_mask]
        status_det = data[status_mask]

        # chọn box có confidence cao nhất trong mỗi nhóm
        if plate_det.numel() > 0:
            plate_det = plate_det[torch.argmax(plate_det[:, 4])].unsqueeze(0)
        else:
            plate_det = torch.empty((0, 6))

        if status_det.numel() > 0:
            status_det = status_det[torch.argmax(status_det[:, 4])].unsqueeze(0)
        else:
            status_det = torch.empty((0, 6))

        # ✅ Kiểm tra empty trước khi index
        plate_result = plate_det[-1] if plate_det.numel() > 0 else torch.empty(6)
        status_result = status_det[-1] if status_det.numel() > 0 else torch.empty(6)

        results.append({"plate": plate_result, "status": status_result})

    return results


def truck_dectect(model, images):
    if len(images) < 1:
        return []
    # ✅ OPTIMIZED: verbose=False để giảm overhead, stream=True cho batch inference
    preds = model.predict(images, verbose=False, stream=False, half=True, device=0)
    post_results = post_processing(preds)

    final_results = []
    for res in post_results:
        plate_ = res.get("plate", [])
        status_ = res.get("status", [])
        plate_bbox = []
        plate_conf = 0.0
        status_conf = 0.0
        status_idx = 0
        status_bbox = []

        if plate_.numel() > 0:
            plate_bbox, plate_conf = plate_[0:4], plate_[4]
        if status_.numel() > 0:
            status_bbox, status_conf, status_idx = status_[0:4], status_[4], status_[-1]

        res = {
            "plate_bbox": plate_bbox,
            "plate_conf": plate_conf,
            "status_bbox": status_bbox,
            "status_conf": status_conf,
            "status_idx": status_idx,
        }
        final_results.append(res)

    return final_results


# yolo_model = str(pathlib.Path(""))
# model = YOLO("track_model/best.pt", task = "detect")
# print(truck_dectect(model, ["images/22L-7067_43334.jpg"]*4))
