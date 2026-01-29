#!/usr/bin/env python3
"""
Micro-test: Prove if pruning (slicing tensors) releases memory
"""
import torch
import gc

def test_pruning_without_explicit_delete():
    """Test memory release when slicing tensors WITHOUT explicit delete"""
    print("Test 1: Pruning WITHOUT explicit delete (current code)")
    
    torch.cuda.empty_cache()
    gc.collect()
    
    # Simulate 500K Gaussians with multiple attributes
    n_gaussians = 500000
    xyz = torch.randn((n_gaussians, 3), device="cuda")
    opacity = torch.randn((n_gaussians, 1), device="cuda")
    scaling = torch.randn((n_gaussians, 3), device="cuda")
    rotation = torch.randn((n_gaussians, 4), device="cuda")
    features = torch.randn((n_gaussians, 48), device="cuda")  # SH features
    
    # Gradient accumulation buffers
    xyz_grad_accum = torch.randn((n_gaussians, 1), device="cuda")
    denom = torch.randn((n_gaussians, 1), device="cuda")
    
    initial_mem = torch.cuda.memory_allocated() / (1024**2)  # MB
    print(f"Initial memory (500K Gaussians): {initial_mem:.2f} MB")
    
    # Simulate pruning: Remove 185K Gaussians (37%)
    prune_mask = torch.rand(n_gaussians, device="cuda") < 0.37
    valid_mask = ~prune_mask
    n_pruned = prune_mask.sum().item()
    print(f"Pruning {n_pruned:,} Gaussians ({n_pruned/n_gaussians*100:.1f}%)")
    
    # Current code: Just reassign with slicing (NO explicit delete)
    xyz = xyz[valid_mask]
    opacity = opacity[valid_mask]
    scaling = scaling[valid_mask]
    rotation = rotation[valid_mask]
    features = features[valid_mask]
    xyz_grad_accum = xyz_grad_accum[valid_mask]
    denom = denom[valid_mask]
    
    gc.collect()
    torch.cuda.empty_cache()
    
    after_prune_mem = torch.cuda.memory_allocated() / (1024**2)
    memory_freed = initial_mem - after_prune_mem
    print(f"After pruning: {after_prune_mem:.2f} MB")
    print(f"Memory freed: {memory_freed:.2f} MB")
    
    expected_freed = n_pruned * (3 + 1 + 3 + 4 + 48 + 1 + 1) * 4 / (1024**2)  # bytes to MB
    print(f"Expected freed: {expected_freed:.2f} MB")
    
    del xyz, opacity, scaling, rotation, features, xyz_grad_accum, denom
    torch.cuda.empty_cache()
    
    return memory_freed, expected_freed

def test_pruning_with_explicit_delete():
    """Test memory release when slicing tensors WITH explicit delete"""
    print("\nTest 2: Pruning WITH explicit delete (proposed fix)")
    
    torch.cuda.empty_cache()
    gc.collect()
    
    # Simulate 500K Gaussians
    n_gaussians = 500000
    xyz = torch.randn((n_gaussians, 3), device="cuda")
    opacity = torch.randn((n_gaussians, 1), device="cuda")
    scaling = torch.randn((n_gaussians, 3), device="cuda")
    rotation = torch.randn((n_gaussians, 4), device="cuda")
    features = torch.randn((n_gaussians, 48), device="cuda")
    xyz_grad_accum = torch.randn((n_gaussians, 1), device="cuda")
    denom = torch.randn((n_gaussians, 1), device="cuda")
    
    initial_mem = torch.cuda.memory_allocated() / (1024**2)
    print(f"Initial memory (500K Gaussians): {initial_mem:.2f} MB")
    
    # Prune 37%
    prune_mask = torch.rand(n_gaussians, device="cuda") < 0.37
    valid_mask = ~prune_mask
    n_pruned = prune_mask.sum().item()
    print(f"Pruning {n_pruned:,} Gaussians ({n_pruned/n_gaussians*100:.1f}%)")
    
    # Proposed fix: Explicit delete before reassigning
    xyz_new = xyz[valid_mask]
    opacity_new = opacity[valid_mask]
    scaling_new = scaling[valid_mask]
    rotation_new = rotation[valid_mask]
    features_new = features[valid_mask]
    xyz_grad_accum_new = xyz_grad_accum[valid_mask]
    denom_new = denom[valid_mask]
    
    # Delete old tensors explicitly
    del xyz, opacity, scaling, rotation, features, xyz_grad_accum, denom
    
    # Reassign
    xyz = xyz_new
    opacity = opacity_new
    scaling = scaling_new
    rotation = rotation_new
    features = features_new
    xyz_grad_accum = xyz_grad_accum_new
    denom = denom_new
    
    gc.collect()
    torch.cuda.empty_cache()
    
    after_prune_mem = torch.cuda.memory_allocated() / (1024**2)
    memory_freed = initial_mem - after_prune_mem
    print(f"After pruning: {after_prune_mem:.2f} MB")
    print(f"Memory freed: {memory_freed:.2f} MB")
    
    expected_freed = n_pruned * (3 + 1 + 3 + 4 + 48 + 1 + 1) * 4 / (1024**2)
    print(f"Expected freed: {expected_freed:.2f} MB")
    
    del xyz, opacity, scaling, rotation, features, xyz_grad_accum, denom
    del xyz_new, opacity_new, scaling_new, rotation_new, features_new, xyz_grad_accum_new, denom_new
    torch.cuda.empty_cache()
    
    return memory_freed, expected_freed

def test_in_place_vs_new_tensor():
    """Test if in-place slicing creates new tensor or references old one"""
    print("\nTest 3: Check if slicing creates new tensor or shares memory")
    
    torch.cuda.empty_cache()
    gc.collect()
    
    # Create original tensor
    original = torch.randn((1000000, 3), device="cuda")
    original_ptr = original.data_ptr()
    initial_mem = torch.cuda.memory_allocated() / (1024**2)
    
    print(f"Original tensor: {original.shape}, ptr={hex(original_ptr)}, mem={initial_mem:.2f} MB")
    
    # Slice it (like pruning does)
    mask = torch.rand(1000000, device="cuda") > 0.5
    sliced = original[mask]
    sliced_ptr = sliced.data_ptr()
    
    print(f"Sliced tensor: {sliced.shape}, ptr={hex(sliced_ptr)}")
    print(f"Same data_ptr? {original_ptr == sliced_ptr}")
    
    # Reassign (like current code)
    original = original[mask]
    after_reassign_mem = torch.cuda.memory_allocated() / (1024**2)
    
    print(f"After reassignment: mem={after_reassign_mem:.2f} MB")
    print(f"Memory freed: {initial_mem - after_reassign_mem:.2f} MB")
    
    return initial_mem - after_reassign_mem

if __name__ == "__main__":
    print("=" * 70)
    print("Memory Leak Investigation: Pruning Test")
    print("=" * 70)
    
    freed_without, expected_without = test_pruning_without_explicit_delete()
    freed_with, expected_with = test_pruning_with_explicit_delete()
    freed_reassign = test_in_place_vs_new_tensor()
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY:")
    print("=" * 70)
    print(f"Test 1 (current code - no explicit delete):")
    print(f"  Freed: {freed_without:.2f} MB / Expected: {expected_without:.2f} MB")
    print(f"  Efficiency: {freed_without/expected_without*100:.1f}%")
    
    print(f"\nTest 2 (proposed fix - explicit delete):")
    print(f"  Freed: {freed_with:.2f} MB / Expected: {expected_with:.2f} MB")
    print(f"  Efficiency: {freed_with/expected_with*100:.1f}%")
    
    print(f"\nTest 3 (in-place reassignment):")
    print(f"  Freed: {freed_reassign:.2f} MB")
    
    print("\n" + "=" * 70)
    if freed_without < expected_without * 0.8:  # Less than 80% freed
        print("⚠️  LEAK CONFIRMED: Current pruning code does NOT free memory properly!")
        print(f"   Expected to free {expected_without:.2f} MB, only freed {freed_without:.2f} MB")
        if freed_with > freed_without * 1.1:  # 10% improvement
            print("✓  FIX WORKS: Explicit delete frees more memory!")
    else:
        print("✓  NO LEAK: Pruning properly frees memory without explicit delete")
    print("=" * 70)
