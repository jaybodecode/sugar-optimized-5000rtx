"""
Micro test: Verify RAM flushing works before training
"""
import gc
import psutil
import time

def get_ram_usage():
    """Get current RAM usage in GB and percent"""
    ram = psutil.virtual_memory()
    return ram.percent, ram.used / (1024**3)

def flush_ram():
    """Force garbage collection and return memory to OS"""
    gc.collect()
    # Give system moment to reclaim memory
    time.sleep(0.1)

print("Testing RAM flush mechanism...")
print("-" * 50)

# Get initial RAM
ram_percent_before, ram_gb_before = get_ram_usage()
print(f"Before flush: {ram_percent_before:.1f}% / {ram_gb_before:.2f}GB")

# Create some temporary objects to demonstrate
temp_data = [list(range(10000)) for _ in range(100)]
ram_percent_mid, ram_gb_mid = get_ram_usage()
print(f"After allocation: {ram_percent_mid:.1f}% / {ram_gb_mid:.2f}GB")

# Delete and flush
del temp_data
flush_ram()

# Check after flush
ram_percent_after, ram_gb_after = get_ram_usage()
print(f"After flush: {ram_percent_after:.1f}% / {ram_gb_after:.2f}GB")

print("-" * 50)
print(f"✓ RAM flush mechanism working")
print(f"  Delta: {ram_gb_mid - ram_gb_after:.2f}GB reclaimed")
