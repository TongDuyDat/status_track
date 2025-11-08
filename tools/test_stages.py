"""
Test từng stage của pipeline riêng lẻ
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from pipelines.track_detect_pipeline import truck_dectect
from pipelines.text_dectect_pipeline import text_detect
from pipelines.text_recognition_pipeline import ocr_inference
from pipelines.model_manager import load_model


def test_truck_detection(image_path):
    """Test truck detection"""
    print(f"\n{'='*60}")
    print(f"🚛 TRUCK DETECTION TEST: {image_path}")
    print("=" * 60)

    # Load model và image
    model = load_model("track_model/best.pt", type="pt")
    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ Cannot load image")
        return None

    print(f"Image shape: {image.shape}")

    # Detect
    results = truck_dectect(model, [image])

    print(f"\nResults: {results}")

    if results and results[0].get("bbox"):
        bbox = results[0]["bbox"]
        conf = results[0]["conf"]
        print(f"✅ Plate detected:")
        print(f"  BBox: {bbox}")
        print(f"  Confidence: {conf:.3f}")
        return image, bbox
    else:
        print(f"⚠️  No plate detected")
        return None


def test_text_detection(image, bbox):
    """Test text detection"""
    print(f"\n{'='*60}")
    print(f"📝 TEXT DETECTION TEST")
    print("=" * 60)

    # Crop plate
    x1, y1, x2, y2 = [int(v) for v in bbox]
    plate_crop = image[y1:y2, x1:x2]

    print(f"Plate crop shape: {plate_crop.shape}")

    # Load model và detect
    model = load_model("text_detection/inference.onnx")
    bboxes = text_detect(model, plate_crop)

    print(f"\nDetected {len(bboxes)} text regions")

    if bboxes:
        print(f"✅ Text regions:")
        for i, bbox_info in enumerate(bboxes, 1):
            bbox_coords = bbox_info[0] if len(bbox_info) > 0 else bbox_info
            print(f"  Region {i}: {bbox_coords}")
        return plate_crop, bboxes
    else:
        print(f"⚠️  No text regions detected")
        return None


def test_ocr(plate_crop, text_bboxes):
    """Test OCR"""
    print(f"\n{'='*60}")
    print(f"🔤 OCR TEST")
    print("=" * 60)

    # Crop text regions
    crops = []
    for bbox_info in text_bboxes:
        bbox_coords = bbox_info[0] if len(bbox_info) > 0 else bbox_info
        x1, y1, x2, y2 = [int(v) for v in bbox_coords[:4]]
        crop = plate_crop[y1:y2, x1:x2]
        crops.append(crop)
        print(f"  Crop shape: {crop.shape}")

    # OCR
    model = load_model(
        "text_recognition/pretrained=parseq-patch16-224_prepost_process_v2.onnx"
    )
    results = ocr_inference(model, crops)

    print(f"\nOCR Results:")
    full_text = ""
    for i, result in enumerate(results, 1):
        text = result.get("text", "")
        conf = result.get("conf", 0)
        print(f"  Crop {i}: '{text}' (conf: {conf:.3f})")
        full_text += text

    print(f"\n✅ Full plate number: '{full_text}'")


def main():
    """Run full test"""
    import sys

    if len(sys.argv) < 2:
        image_path = "images/22L-7067_43334.jpg"
        print(f"Using default image: {image_path}")
    else:
        image_path = sys.argv[1]

    print(f"\n{'='*60}")
    print(f"🧪 STAGE-BY-STAGE PIPELINE TEST")
    print(f"{'='*60}")

    # Stage 1: Truck detection
    result = test_truck_detection(image_path)
    if not result:
        print(f"\n❌ Cannot proceed - no plate detected")
        return

    image, bbox = result

    # Stage 2: Text detection
    result = test_text_detection(image, bbox)
    if not result:
        print(f"\n❌ Cannot proceed - no text regions detected")
        return

    plate_crop, text_bboxes = result

    # Stage 3: OCR
    test_ocr(plate_crop, text_bboxes)

    print(f"\n{'='*60}")
    print(f"✅ ALL STAGES COMPLETED")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
