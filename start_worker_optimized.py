"""
Start optimized worker with batch configuration
"""

import os
import sys


def start_worker():
    print("=" * 60)
    print("🚀 STARTING OPTIMIZED WORKER")
    print("=" * 60)
    # Set optimized environment variables
    os.environ["WORKER_BATCH_SIZE"] = "8"
    os.environ["BATCH_TIMEOUT"] = "0.03"
    os.environ["MAX_CONCURRENT_BATCHES"] = "3"
    os.environ["MAX_WAIT_TIME"] = "0.1"

    # Set logging
    os.environ["LOG_LEVEL"] = "DEBUG"

    print("\n📋 Configuration:")
    print(f"  WORKER_BATCH_SIZE: 8")
    print(f"  BATCH_TIMEOUT: 30ms")
    print(f"  MAX_CONCURRENT_BATCHES: 3")
    print(f"  MAX_WAIT_TIME: 100ms")
    print(f"  LOG_LEVEL: DEBUG")
    print("\n" + "=" * 60)
    print("💡 Look for these log messages:")
    print("  - 'Processing batch of 8 tasks'")
    print("  - 'Collected X tasks in Y.YYs'")
    print("  - BatchManager logs with batch_size > 1")
    print("=" * 60)
    print()
    # Import and run worker
    from worker.image_processor import main
    import asyncio

    asyncio.run(main())


if __name__ == "__main__":
    start_worker()
