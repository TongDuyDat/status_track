"""
Real-time memory monitoring script
Hiển thị RAM, GPU, và Batch Manager stats
"""

import requests
import time
import sys
from datetime import datetime

API_URL = "http://localhost:8000"
REFRESH_INTERVAL = 2  # seconds


def clear_screen():
    """Clear terminal screen"""
    import os

    os.system("cls" if os.name == "nt" else "clear")


def format_bytes(bytes_val):
    """Format bytes to GB"""
    return f"{bytes_val:.2f}GB"


def get_color(percent):
    """Get color based on percentage"""
    if percent < 60:
        return "\033[92m"  # Green
    elif percent < 80:
        return "\033[93m"  # Yellow
    else:
        return "\033[91m"  # Red


def reset_color():
    return "\033[0m"


def print_bar(label, value, max_val, width=40):
    """Print progress bar"""
    percent = (value / max_val) * 100 if max_val > 0 else 0
    filled = int(width * value / max_val) if max_val > 0 else 0
    bar = "█" * filled + "░" * (width - filled)

    color = get_color(percent)
    print(f"{label:20s} {color}[{bar}] {percent:6.2f}%{reset_color()}")


def monitor_loop():
    """Main monitoring loop"""
    print("🔍 Starting memory monitor...")
    print("Press Ctrl+C to stop\n")

    iteration = 0

    try:
        while True:
            clear_screen()

            print("=" * 80)
            print(f"🧠 MEMORY MONITOR - {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 80)

            # Get memory stats
            try:
                mem_resp = requests.get(f"{API_URL}/api/monitor/memory", timeout=2)
                mem_data = mem_resp.json()

                # RAM Stats
                print("\n📊 RAM Usage:")
                ram = mem_data["ram"]
                print_bar("RAM", ram["used_gb"], ram["total_gb"])
                print(
                    f"    Used: {format_bytes(ram['used_gb'])} / {format_bytes(ram['total_gb'])}"
                )
                print(f"    Available: {format_bytes(ram['available_gb'])}")

                # GPU Stats
                if mem_data["gpu"]["available"]:
                    print("\n🎮 GPU Memory:")
                    gpu = mem_data["gpu"]
                    print(f"    Device: {gpu['device_name']}")
                    print(f"    Allocated: {format_bytes(gpu['allocated_gb'])}")
                    print(f"    Reserved: {format_bytes(gpu['reserved_gb'])}")
                    print(f"    Max Allocated: {format_bytes(gpu['max_allocated_gb'])}")
                else:
                    print("\n🎮 GPU: Not available")

            except requests.RequestException as e:
                print(f"\n❌ Error fetching memory stats: {e}")

            # Get batch manager stats
            try:
                batch_resp = requests.get(
                    f"{API_URL}/api/monitor/batch-managers", timeout=2
                )
                batch_data = batch_resp.json()

                print("\n📦 Batch Managers:")
                for name, stats in batch_data.items():
                    display_name = name.replace("_", " ").title()
                    queue_len = stats["queue_length"]
                    batch_size = stats["batch_size"]
                    latency = stats["last_latency"]
                    mem_usage = stats["memory_usage"] * 100

                    print(f"\n  {display_name}:")
                    print(f"    Queue Length: {queue_len}")
                    print(f"    Batch Size: {batch_size}")
                    print(f"    Latency: {latency:.3f}s")
                    print(f"    Memory Usage: {mem_usage:.1f}%")

            except requests.RequestException as e:
                print(f"\n❌ Error fetching batch stats: {e}")

            # Footer
            print("\n" + "=" * 80)
            print(
                f"Iteration: {iteration} | Refresh: {REFRESH_INTERVAL}s | Press Ctrl+C to stop"
            )
            print("=" * 80)

            iteration += 1
            time.sleep(REFRESH_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped")
        sys.exit(0)


def main():
    """Entry point"""
    # Check if API is accessible
    try:
        resp = requests.get(f"{API_URL}/api/monitor/health", timeout=5)
        if resp.status_code != 200:
            print(f"❌ API not accessible at {API_URL}")
            print("Make sure FastAPI server is running:")
            print("  uvicorn main:app --reload")
            sys.exit(1)
    except requests.RequestException:
        print(f"❌ Cannot connect to {API_URL}")
        print("Make sure FastAPI server is running:")
        print("  uvicorn main:app --reload")
        sys.exit(1)

    monitor_loop()


if __name__ == "__main__":
    main()
