#!/usr/bin/env python3
"""
Quick API health check
"""

import asyncio
import httpx
import sys

API_URL = "http://localhost:8000"


async def check_health():
    """Check if API server is running"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{API_URL}/api/monitor/health")

            if response.status_code == 200:
                print("✅ API server is running")
                return True
            else:
                print(f"⚠️  API server returned status {response.status_code}")
                return False
    except httpx.ConnectError:
        print("❌ Cannot connect to API server")
        print(f"   Make sure server is running at {API_URL}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def check_routes():
    """Check if required routes exist"""
    print("\n🔍 Checking API routes...")

    routes = {
        "/api/monitor/health": "GET",
        "/api/monitor/memory": "GET",
        "/api/monitor/batch-managers": "GET",
        "/api/upload": "POST (multipart)",
        "/api/upload/image": "POST (JSON base64)",
    }

    async with httpx.AsyncClient(timeout=5.0) as client:
        # Check health
        try:
            r = await client.get(f"{API_URL}/api/monitor/health")
            if r.status_code == 200:
                print("  ✅ /api/monitor/health")
            else:
                print(f"  ⚠️  /api/monitor/health - {r.status_code}")
        except:
            print("  ❌ /api/monitor/health")

        # Check memory
        try:
            r = await client.get(f"{API_URL}/api/monitor/memory")
            if r.status_code == 200:
                print("  ✅ /api/monitor/memory")
            else:
                print(f"  ⚠️  /api/monitor/memory - {r.status_code}")
        except:
            print("  ❌ /api/monitor/memory")

        # Check batch-managers
        try:
            r = await client.get(f"{API_URL}/api/monitor/batch-managers")
            if r.status_code == 200:
                print("  ✅ /api/monitor/batch-managers")
            else:
                print(f"  ⚠️  /api/monitor/batch-managers - {r.status_code}")
        except:
            print("  ❌ /api/monitor/batch-managers")

        # Check upload/image (base64)
        try:
            r = await client.post(
                f"{API_URL}/api/upload/image",
                json={"image": "dGVzdA=="},  # "test" in base64
            )
            if r.status_code == 200:
                print("  ✅ /api/upload/image (JSON base64)")
            else:
                print(f"  ⚠️  /api/upload/image - {r.status_code}")
                if r.status_code == 404:
                    print("     ❌ Route not found! Restart API server.")
        except Exception as e:
            print(f"  ❌ /api/upload/image - {e}")


async def check_redis():
    """Check if Redis is accessible"""
    print("\n🔍 Checking Redis connection...")

    try:
        from utils.redis_utils import redis_client

        # Try ping
        redis_client.ping()
        print("  ✅ Redis is accessible")

        # Check queue
        queue_len = redis_client.llen("task_queue")
        print(f"  📊 Current queue length: {queue_len}")

        return True
    except Exception as e:
        print(f"  ❌ Redis error: {e}")
        return False


async def check_worker():
    """Check if worker is processing tasks"""
    print("\n🔍 Checking worker status...")

    try:
        from utils.redis_utils import redis_client

        # Push test task
        test_task = {"task_id": "health_check_test", "image_bytes": "00"}  # dummy

        import json

        redis_client.lpush("task_queue", json.dumps(test_task))
        print("  📤 Pushed test task to queue")

        # Wait and check if processed
        await asyncio.sleep(2)

        queue_len = redis_client.llen("task_queue")
        if queue_len == 0:
            print("  ✅ Worker is processing tasks (queue cleared)")
        else:
            print(f"  ⚠️  Worker may not be running (queue: {queue_len})")
            print("     Start worker: python start_staged_worker.py")

    except Exception as e:
        print(f"  ❌ Error: {e}")


def check_onnx_gpu():
    """Check if ONNX models are using GPU"""
    print("\n🔍 Checking ONNX Runtime GPU support...")

    try:
        import onnxruntime as ort

        available = ort.get_available_providers()
        has_tensorrt = "TensorrtExecutionProvider" in available
        has_cuda = "CUDAExecutionProvider" in available

        print(f"  Available providers: {', '.join(available)}")

        if has_tensorrt:
            print("  🚀 TensorRT: Available (BEST performance)")
        elif has_cuda:
            print("  ✅ CUDA: Available (Good performance)")
        else:
            print("  ❌ GPU: NOT AVAILABLE")
            print("     ⚠️  Models will run on CPU (VERY SLOW!)")
            print("     Fix: python fix_onnx_gpu.py")
            return False

        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


async def main():
    print("=" * 70)
    print("🏥 SYSTEM HEALTH CHECK")
    print("=" * 70)

    # Check API
    api_ok = await check_health()

    if not api_ok:
        print("\n⚠️  API server is not running!")
        print("\n📋 Start API server:")
        print("  python main.py")
        sys.exit(1)

    # Check routes
    await check_routes()

    # Check Redis
    redis_ok = await check_redis()

    if not redis_ok:
        print("\n⚠️  Redis is not accessible!")
        print("\n📋 Start Redis:")
        print("  redis-server")
        sys.exit(1)

    # Check worker
    await check_worker()

    # Check ONNX GPU support
    onnx_gpu_ok = check_onnx_gpu()

    print("\n" + "=" * 70)
    print("✅ SYSTEM STATUS")
    print("=" * 70)

    if api_ok and redis_ok:
        print("✅ Ready to run tests!")

        if not onnx_gpu_ok:
            print("\n⚠️  ONNX GPU support missing!")
            print("   Models will run on CPU (very slow)")
            print("   Fix: python fix_onnx_gpu.py")
            print("   Or check: python check_onnx_gpu.py")

        print("\n📊 Run GPU load test:")
        print("  python tests/test_gpu_load.py")
        print("\n📈 Monitor GPU:")
        print("  python monitor_gpu.py")
        print("\n🔍 Check ONNX GPU:")
        print("  python check_onnx_gpu.py")
    else:
        print("⚠️  Some components are not ready")


if __name__ == "__main__":
    asyncio.run(main())
