"""
Debug script - Test pipeline trực tiếp không qua Redis
"""

import asyncio
import cv2
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.pipeline_async import pipeline_async


async def test_single_image(image_path):
    """Test 1 ảnh"""
    print(f"📷 Testing image: {image_path}")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Cannot load image: {image_path}")
        return

    print(f"  Image shape: {image.shape}")

    # Process
    try:
        print("  🔄 Processing...")
        results = await pipeline_async([image])

        if not results:
            print("  ⚠️  No results returned (no plates detected)")
        else:
            print(f"  ✅ Results: {len(results)} plate(s)")
            for i, result in enumerate(results, 1):
                print(f"\n  Plate {i}:")
                print(f"    Number: {result.get('plate_number', 'N/A')}")
                print(f"    BBox: {result.get('plate_bbox', [])}")
                print(f"    Truck Conf: {result.get('truck_conf', 0):.3f}")
                print(f"    OCR Conf: {result.get('ocr_conf', 0):.3f}")

    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback

        traceback.print_exc()


async def test_multiple_images(image_dir="images"):
    """Test nhiều ảnh"""
    images_path = Path(image_dir)
    image_files = list(images_path.glob("22L-*.jpg"))[:5]  # Test 5 ảnh đầu

    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return

    print(f"🧪 Testing {len(image_files)} images...\n")

    for img_path in image_files:
        await test_single_image(str(img_path))
        print("\n" + "=" * 60 + "\n")


async def test_empty_image():
    """Test ảnh không có plate"""
    import numpy as np

    print("🧪 Testing empty image (no plate)...")

    # Tạo ảnh trống
    dummy_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)

    try:
        results = await pipeline_async([dummy_image])
        print(f"  Results: {results}")
        if not results:
            print("  ✅ Correctly returned empty list")
        else:
            print("  ⚠️  Unexpected results on empty image")
    except Exception as e:
        print(f"  ❌ Error: {e}")


async def main():
    """Run all tests"""
    print("=" * 60)
    print("🔧 PIPELINE DEBUG TOOL")
    print("=" * 60 + "\n")

    # Test 1: Empty image
    await test_empty_image()
    print("\n" + "=" * 60 + "\n")

    # Test 2: Real images
    await test_multiple_images()


if __name__ == "__main__":
    asyncio.run(main())
