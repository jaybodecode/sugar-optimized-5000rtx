#!/usr/bin/env python
"""Test script to verify lazy image loading implementation.

This tests that:
1. Images are not loaded into VRAM until accessed
2. LRU cache keeps only last 10 images
3. Cache clearing works properly
"""

import torch
import sys
sys.path.append('.')

from sugar_scene.gs_model import GaussianSplattingWrapper

def get_vram_usage_gb():
    """Get current VRAM usage in GB."""
    return torch.cuda.memory_allocated() / 1024**3

print("=" * 70)
print("Testing Lazy Image Loading Implementation")
print("=" * 70)

# Test paths
source_path = "/mnt/c/Users/techm/Downloads/TRAIN_DATASETS/360_v2/garden"
gs_output_path = "/mnt/c/Users/techm/Downloads/TRAIN_DATASETS/360_v2/garden_output/20260125_024541/"

print("\n1. Loading GaussianSplattingWrapper WITHOUT loading images...")
initial_vram = get_vram_usage_gb()
print(f"   Initial VRAM: {initial_vram:.2f} GB")

# Load wrapper (should NOT load images yet)
nerfmodel = GaussianSplattingWrapper(
    source_path=source_path,
    output_path=gs_output_path,
    iteration_to_load=7000,
    eval_split=True,
    load_gt_images=True,  # Enable lazy loading
)

after_load_vram = get_vram_usage_gb()
print(f"   After loading wrapper: {after_load_vram:.2f} GB")
print(f"   VRAM increase: {after_load_vram - initial_vram:.2f} GB")
print(f"   ✓ Should be ~8-9GB (Gaussians + transforms), NOT ~13-14GB (with images)")

print("\n2. Accessing first image (should load to cache)...")
before_access = get_vram_usage_gb()
img = nerfmodel.get_gt_image(camera_indices=0)
after_access = get_vram_usage_gb()
print(f"   Before: {before_access:.2f} GB, After: {after_access:.2f} GB")
print(f"   Increase: {after_access - before_access:.2f} GB (~0.03GB for 1 image)")
print(f"   Image shape: {img.shape}")

print("\n3. Accessing 15 images (should keep only last 10 in cache)...")
before_multi = get_vram_usage_gb()
for i in range(15):
    _ = nerfmodel.get_gt_image(camera_indices=i)
after_multi = get_vram_usage_gb()
print(f"   Before: {before_multi:.2f} GB, After: {after_multi:.2f} GB")
print(f"   Increase: {after_multi - before_multi:.2f} GB (~0.3GB for 10 images)")

print("\n4. Clearing cache...")
from sugar_scene.cameras import GSCamera
before_clear = get_vram_usage_gb()
GSCamera.clear_image_cache()
after_clear = get_vram_usage_gb()
print(f"   Before: {before_clear:.2f} GB, After: {after_clear:.2f} GB")
print(f"   Freed: {before_clear - after_clear:.2f} GB")

print("\n5. Re-accessing same image (should reload from disk)...")
before_reload = get_vram_usage_gb()
img2 = nerfmodel.get_gt_image(camera_indices=0)
after_reload = get_vram_usage_gb()
print(f"   Before: {before_reload:.2f} GB, After: {after_reload:.2f} GB")
print(f"   Increase: {after_reload - before_reload:.2f} GB")
print(f"   Image matches: {torch.allclose(img, img2, atol=1e-5)}")

print("\n" + "=" * 70)
print("✓ Lazy loading test complete!")
print("=" * 70)
print("\nExpected VRAM usage:")
print("  - Old method: ~18-19GB (all images loaded)")
print("  - New method: ~13-14GB (only 10 images cached)")
print("  - Savings: ~5GB")
