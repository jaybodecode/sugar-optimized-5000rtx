#!/usr/bin/env python3
"""Test pynvml library compatibility with PyTorch CUDA."""

import sys
import torch

print("=" * 60)
print("Testing pynvml compatibility")
print("=" * 60)

# Test 1: Basic pynvml import and init
print("\n[Test 1] Importing pynvml...")
try:
    import pynvml
    print("✓ pynvml imported")
except ImportError as e:
    print(f"✗ Failed to import pynvml: {e}")
    sys.exit(1)

print("\n[Test 2] Initializing pynvml...")
try:
    pynvml.nvmlInit()
    print("✓ pynvml initialized")
except Exception as e:
    print(f"✗ Failed to initialize pynvml: {e}")
    sys.exit(1)

# Test 3: Get GPU handle
print("\n[Test 3] Getting GPU handle...")
try:
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    print("✓ Got GPU handle")
except Exception as e:
    print(f"✗ Failed to get GPU handle: {e}")
    pynvml.nvmlShutdown()
    sys.exit(1)

# Test 4: Query GPU info
print("\n[Test 4] Querying GPU info...")
try:
    name = pynvml.nvmlDeviceGetName(handle)
    print(f"✓ GPU Name: {name}")
    
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    print(f"✓ GPU Utilization: {util.gpu}%")
    
    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    print(f"✓ GPU Temperature: {temp}°C")
    
    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
    print(f"✓ GPU Power: {power:.1f}W")
except Exception as e:
    print(f"✗ Failed to query GPU: {e}")
    pynvml.nvmlShutdown()
    sys.exit(1)

# Test 5: PyTorch CUDA init
print("\n[Test 5] Initializing PyTorch CUDA...")
try:
    device = torch.device("cuda")
    print(f"✓ PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"✓ PyTorch CUDA device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"✗ PyTorch CUDA init failed: {e}")
    pynvml.nvmlShutdown()
    sys.exit(1)

# Test 6: Create tensor and query pynvml
print("\n[Test 6] Creating PyTorch tensor + pynvml query...")
try:
    # Allocate some VRAM
    x = torch.randn(1000, 1000, device="cuda")
    allocated = torch.cuda.memory_allocated() / (1024**3)
    print(f"✓ Allocated {allocated:.2f} GB VRAM")
    
    # Query with pynvml
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    print(f"✓ pynvml query successful: {util.memory}% memory util")
    
    del x
    torch.cuda.empty_cache()
    print("✓ Freed tensor and cache")
except Exception as e:
    print(f"✗ Tensor + pynvml query failed: {e}")
    pynvml.nvmlShutdown()
    sys.exit(1)

# Test 7: Repeated queries (simulate training loop)
print("\n[Test 7] Repeated pynvml queries (100 iterations)...")
try:
    for i in range(100):
        x = torch.randn(100, 100, device="cuda")
        
        if i % 10 == 0:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        
        del x
        
        if i % 20 == 0:
            torch.cuda.empty_cache()
    
    print("✓ 100 iterations completed successfully")
except Exception as e:
    print(f"✗ Repeated queries failed at iteration {i}: {e}")
    pynvml.nvmlShutdown()
    sys.exit(1)

# Test 8: Cleanup
print("\n[Test 8] Cleaning up...")
try:
    torch.cuda.empty_cache()
    pynvml.nvmlShutdown()
    print("✓ Cleanup successful")
except Exception as e:
    print(f"✗ Cleanup failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED - pynvml is compatible!")
print("=" * 60)
