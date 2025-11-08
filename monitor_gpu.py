#!/usr/bin/env python3
"""
Real-time GPU monitoring tool
Hiển thị GPU utilization, memory, batch sizes
"""

import asyncio
import time
import os
import sys
import subprocess
from datetime import datetime

try:
    import httpx

    HAS_HTTPX = True
except:
    HAS_HTTPX = False


def get_gpu_stats():
    """Get GPU stats from nvidia-smi"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )

        if result.returncode == 0:
            gpu_util, mem_used, mem_total = result.stdout.strip().split(",")
            return {
                "gpu_util": int(gpu_util.strip()),
                "mem_used": int(mem_used.strip()),
                "mem_total": int(mem_total.strip()),
                "mem_percent": round(
                    int(mem_used.strip()) / int(mem_total.strip()) * 100, 1
                ),
            }
    except Exception as e:
        pass

    return None


async def get_pipeline_stats():
    """Get pipeline stats from API"""
    if not HAS_HTTPX:
        return None

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(
                "http://localhost:8000/api/monitor/batch-managers"
            )
            if response.status_code == 200:
                return response.json()
    except:
        pass

    return None


def format_bar(value, max_value, width=40):
    """Format a progress bar"""
    filled = int(value / max_value * width)
    bar = "█" * filled + "░" * (width - filled)
    return bar


async def monitor_loop(interval=2):
    """Main monitoring loop"""
    print("=" * 100)
    print("🔍 GPU UTILIZATION MONITOR")
    print("=" * 100)
    print("\nPress Ctrl+C to stop\n")

    iteration = 0

    while True:
        try:
            iteration += 1
            timestamp = datetime.now().strftime("%H:%M:%S")

            # Get GPU stats
            gpu_stats = get_gpu_stats()

            # Get pipeline stats
            pipeline_stats = await get_pipeline_stats()

            # Clear screen (Windows compatible)
            if iteration > 1:
                # Move cursor up
                print(f"\033[{15}A", end="")

            print("=" * 100)
            print(f"⏰ {timestamp} | Iteration #{iteration}")
            print("=" * 100)

            # GPU Stats
            if gpu_stats:
                gpu_util = gpu_stats["gpu_util"]
                mem_percent = gpu_stats["mem_percent"]

                # Color coding
                if gpu_util < 30:
                    color = "\033[91m"  # Red (underutilized)
                    status = "⚠️  UNDERUTILIZED"
                elif gpu_util < 60:
                    color = "\033[93m"  # Yellow
                    status = "⚡ MODERATE"
                elif gpu_util < 80:
                    color = "\033[92m"  # Green
                    status = "✅ GOOD"
                else:
                    color = "\033[96m"  # Cyan
                    status = "🚀 EXCELLENT"

                print(f"\n🎮 GPU UTILIZATION: {color}{gpu_util}%\033[0m {status}")
                print(f"   {format_bar(gpu_util, 100)} {gpu_util}%")

                print(f"\n💾 GPU MEMORY: {mem_percent}%")
                print(
                    f"   {format_bar(mem_percent, 100)} {gpu_stats['mem_used']}MB / {gpu_stats['mem_total']}MB"
                )
            else:
                print(
                    "\n🎮 GPU UTILIZATION: \033[91m[ERROR: nvidia-smi not available]\033[0m"
                )

            # Pipeline Stats
            if pipeline_stats:
                print("\n📊 BATCH MANAGER STATS:")

                # Truck detector
                truck = pipeline_stats.get("truck_detector", {})
                print(
                    f"   🚛 Truck:  batch_size={truck.get('current_batch_size', 0):2d} | "
                    f"queue={truck.get('queue_length', 0):3d} | "
                    f"latency={truck.get('avg_latency', 0):.3f}s"
                )

                # Text detector
                text = pipeline_stats.get("text_detector", {})
                print(
                    f"   📝 Text:   batch_size={text.get('current_batch_size', 0):2d} | "
                    f"queue={text.get('queue_length', 0):3d} | "
                    f"latency={text.get('avg_latency', 0):.3f}s"
                )

                # OCR
                ocr = pipeline_stats.get("ocr", {})
                print(
                    f"   🔤 OCR:    batch_size={ocr.get('current_batch_size', 0):2d} | "
                    f"queue={ocr.get('queue_length', 0):3d} | "
                    f"latency={ocr.get('avg_latency', 0):.3f}s"
                )

                # Pipeline scheduler (if staged mode)
                scheduler = pipeline_stats.get("pipeline_scheduler")
                if scheduler:
                    print(f"\n🔄 PIPELINE SCHEDULER (Staged Mode):")
                    print(
                        f"   Stage 1: {scheduler.get('avg_stage1_time', 0):.3f}s avg | "
                        f"queue={scheduler.get('queue1_size', 0)}"
                    )
                    print(
                        f"   Stage 2: {scheduler.get('avg_stage2_time', 0):.3f}s avg | "
                        f"queue={scheduler.get('queue2_size', 0)}"
                    )
                    print(
                        f"   Stage 3: {scheduler.get('avg_stage3_time', 0):.3f}s avg | "
                        f"queue={scheduler.get('queue3_size', 0)}"
                    )
            else:
                print("\n📊 BATCH MANAGER STATS: \033[93m[API not available]\033[0m")

            # Recommendations
            if gpu_stats:
                print("\n💡 RECOMMENDATIONS:")
                if gpu_util < 30:
                    print("   ⚠️  GPU severely underutilized!")
                    print("   → Increase WORKER_BATCH_SIZE (current: 16 → try 32)")
                    print("   → Increase MAX_CONCURRENT_BATCHES (current: 10 → try 20)")
                    print("   → Check if batch managers have min_batch too low")
                elif gpu_util < 60:
                    print("   ⚡ GPU could be utilized better")
                    print("   → Increase WORKER_BATCH_SIZE to 24-32")
                    print("   → Increase load (more concurrent requests)")
                elif gpu_util < 80:
                    print("   ✅ Good utilization! Fine-tune for peak performance")
                    print("   → Monitor batch_size in logs")
                    print("   → Ensure queue_length > 0 consistently")
                else:
                    print("   🚀 Excellent! GPU is being utilized efficiently")
                    print("   → Current settings are optimal")

            print("=" * 100)

            await asyncio.sleep(interval)

        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring stopped")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            await asyncio.sleep(interval)


def main():
    """Entry point"""

    # Check dependencies
    if not HAS_HTTPX:
        print("⚠️  Warning: httpx not installed. Install with: pip install httpx")
        print("   (Pipeline stats will not be available)\n")

    # Check nvidia-smi
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, timeout=2)
    except:
        print("❌ Error: nvidia-smi not found. Make sure NVIDIA drivers are installed.")
        return

    # Check API
    if HAS_HTTPX:
        import httpx

        try:
            response = httpx.get("http://localhost:8000/api/monitor/health", timeout=2)
            if response.status_code != 200:
                print("⚠️  Warning: API server not responding at http://localhost:8000")
                print("   Make sure the server is running: python main.py\n")
        except:
            print("⚠️  Warning: Cannot connect to API server")
            print("   Make sure the server is running: python main.py\n")

    # Run monitor
    try:
        asyncio.run(monitor_loop(interval=2))
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")


if __name__ == "__main__":
    main()
