#!/usr/bin/env python3
"""Test pynvml with custom CUDA rasterizer loaded."""

import sys
import torch

print("=" * 60)
print("Testing pynvml with custom CUDA modules")
print("=" * 60)

# Test 1: Import pynvml BEFORE custom CUDA modules
print("\n[Test 1] Import pynvml FIRST (before rasterizer)...")
try:
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    print(f"✓ pynvml works: GPU {util.gpu}%")
except Exception as e:
    print(f"✗ pynvml failed: {e}")
    sys.exit(1)

# Test 2: Import custom CUDA rasterizer
print("\n[Test 2] Loading custom CUDA rasterizer...")
try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    print("✓ Rasterizer imported")
except Exception as e:
    print(f"✗ Rasterizer import failed: {e}")
    pynvml.nvmlShutdown()
    sys.exit(1)

# Test 3: Query pynvml after rasterizer loaded
print("\n[Test 3] Query pynvml AFTER rasterizer loaded...")
try:
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    print(f"✓ pynvml still works: GPU {util.gpu}%, {temp}°C")
except Exception as e:
    print(f"✗ pynvml query failed: {e}")
    pynvml.nvmlShutdown()
    sys.exit(1)

# Test 4: Create tensors and query
print("\n[Test 4] Create tensors + query pynvml...")
try:
    x = torch.randn(1000, 1000, device="cuda")
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    print(f"✓ Works with tensors: GPU {util.gpu}%")
    del x
except Exception as e:
    print(f"✗ Failed with tensors: {e}")
    pynvml.nvmlShutdown()
    sys.exit(1)

# Test 5: empty_cache + pynvml
print("\n[Test 5] torch.cuda.empty_cache() + pynvml...")
try:
    x = torch.randn(1000, 1000, device="cuda")
    del x
    torch.cuda.empty_cache()
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    print(f"✓ Works after empty_cache: GPU {util.gpu}%")
except Exception as e:
    print(f"✗ Failed after empty_cache: {e}")
    pynvml.nvmlShutdown()
    sys.exit(1)

# Test 6: Shutdown and re-init
print("\n[Test 6] Shutdown + re-init pynvml...")
try:
    pynvml.nvmlShutdown()
    print("✓ Shutdown successful")
    
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    print(f"✓ Re-init successful: GPU {util.gpu}%")
except Exception as e:
    print(f"✗ Shutdown/re-init failed: {e}")
    sys.exit(1)

# Cleanup
print("\n[Test 7] Final cleanup...")
try:
    torch.cuda.empty_cache()
    pynvml.nvmlShutdown()
    print("✓ Cleanup successful")
except Exception as e:
    print(f"✗ Cleanup failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED!")
print("=" * 60)
