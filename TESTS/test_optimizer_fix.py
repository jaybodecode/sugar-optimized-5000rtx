#!/usr/bin/env python3
"""
Test different approaches to fix the optimizer state leak
"""
import torch
import torch.nn as nn
import gc

def get_memory():
    """Get current GPU memory in MB"""
    return torch.cuda.memory_allocated() / (1024**2)

def test_current_approach():
    """Current broken approach from _prune_optimizer"""
    print("Test 1: CURRENT APPROACH (broken)")
    print("-" * 60)
    
    torch.cuda.empty_cache()
    gc.collect()
    
    # Create parameter with optimizer state
    param = nn.Parameter(torch.randn(100000, 10, device="cuda"))
    optimizer = torch.optim.Adam([param], lr=0.001)
    
    # Initialize optimizer state
    loss = param.sum()
    loss.backward()
    optimizer.step()
    
    initial_mem = get_memory()
    print(f"Initial memory: {initial_mem:.2f} MB")
    
    # Current approach: Slice and reassign in state dict
    state = optimizer.state[param]
    mask = torch.arange(50000, device="cuda")  # Keep first 50%
    
    # THIS IS THE PROBLEM - modifying state dict in place
    state["exp_avg"] = state["exp_avg"][mask]
    state["exp_avg_sq"] = state["exp_avg_sq"][mask]
    
    del optimizer.state[param]
    
    new_param = nn.Parameter(param.data[mask].requires_grad_(True))
    optimizer.state[new_param] = state
    
    del param
    gc.collect()
    torch.cuda.empty_cache()
    
    after_mem = get_memory()
    print(f"After pruning: {after_mem:.2f} MB")
    print(f"Change: {after_mem - initial_mem:+.2f} MB")
    print(f"Expected: ~-15 MB (should FREE memory)")
    
    return after_mem - initial_mem

def test_fix_approach_1():
    """Fix 1: Create NEW state dict instead of modifying"""
    print("\nTest 2: FIX #1 - Create new state dict")
    print("-" * 60)
    
    torch.cuda.empty_cache()
    gc.collect()
    
    param = nn.Parameter(torch.randn(100000, 10, device="cuda"))
    optimizer = torch.optim.Adam([param], lr=0.001)
    
    loss = param.sum()
    loss.backward()
    optimizer.step()
    
    initial_mem = get_memory()
    print(f"Initial memory: {initial_mem:.2f} MB")
    
    # FIX: Create NEW state dict, don't reuse old one
    old_state = optimizer.state[param]
    mask = torch.arange(50000, device="cuda")
    
    # Create completely new state dict
    new_state = {
        "step": old_state["step"],
        "exp_avg": old_state["exp_avg"][mask].clone(),
        "exp_avg_sq": old_state["exp_avg_sq"][mask].clone()
    }
    
    # Delete old state BEFORE creating new param
    del optimizer.state[param]
    del old_state  # Explicitly delete reference
    
    new_param = nn.Parameter(param.data[mask].requires_grad_(True))
    optimizer.state[new_param] = new_state
    
    del param
    gc.collect()
    torch.cuda.empty_cache()
    
    after_mem = get_memory()
    print(f"After pruning: {after_mem:.2f} MB")
    print(f"Change: {after_mem - initial_mem:+.2f} MB")
    
    return after_mem - initial_mem

def test_fix_approach_2():
    """Fix 2: Delete old tensors explicitly before creating new ones"""
    print("\nTest 3: FIX #2 - Explicit tensor deletion")
    print("-" * 60)
    
    torch.cuda.empty_cache()
    gc.collect()
    
    param = nn.Parameter(torch.randn(100000, 10, device="cuda"))
    optimizer = torch.optim.Adam([param], lr=0.001)
    
    loss = param.sum()
    loss.backward()
    optimizer.step()
    
    initial_mem = get_memory()
    print(f"Initial memory: {initial_mem:.2f} MB")
    
    state = optimizer.state[param]
    mask = torch.arange(50000, device="cuda")
    
    # Slice first
    new_exp_avg = state["exp_avg"][mask]
    new_exp_avg_sq = state["exp_avg_sq"][mask]
    
    # Delete old tensors explicitly
    del state["exp_avg"]
    del state["exp_avg_sq"]
    
    # Assign new tensors
    state["exp_avg"] = new_exp_avg
    state["exp_avg_sq"] = new_exp_avg_sq
    
    del optimizer.state[param]
    
    new_param = nn.Parameter(param.data[mask].requires_grad_(True))
    optimizer.state[new_param] = state
    
    del param
    gc.collect()
    torch.cuda.empty_cache()
    
    after_mem = get_memory()
    print(f"After pruning: {after_mem:.2f} MB")
    print(f"Change: {after_mem - initial_mem:+.2f} MB")
    
    return after_mem - initial_mem

def test_fix_approach_3():
    """Fix 3: Clear all optimizer state and rebuild"""
    print("\nTest 4: FIX #3 - Rebuild optimizer state from scratch")
    print("-" * 60)
    
    torch.cuda.empty_cache()
    gc.collect()
    
    param = nn.Parameter(torch.randn(100000, 10, device="cuda"))
    optimizer = torch.optim.Adam([param], lr=0.001)
    
    loss = param.sum()
    loss.backward()
    optimizer.step()
    
    initial_mem = get_memory()
    print(f"Initial memory: {initial_mem:.2f} MB")
    
    mask = torch.arange(50000, device="cuda")
    
    # Clear ALL optimizer state
    optimizer.state.clear()
    
    # Create new parameter
    new_param = nn.Parameter(param.data[mask].requires_grad_(True))
    optimizer.param_groups[0]['params'][0] = new_param
    
    # Let optimizer reinitialize state on next step (lazy)
    # This means we lose momentum, but guarantee no leak
    
    del param
    gc.collect()
    torch.cuda.empty_cache()
    
    after_mem = get_memory()
    print(f"After pruning: {after_mem:.2f} MB")
    print(f"Change: {after_mem - initial_mem:+.2f} MB")
    print(f"Note: Optimizer state will be reinitialized on next step")
    
    return after_mem - initial_mem

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Optimizer State Leak Fixes")
    print("=" * 70)
    
    current = test_current_approach()
    fix1 = test_fix_approach_1()
    fix2 = test_fix_approach_2()
    fix3 = test_fix_approach_3()
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY:")
    print("=" * 70)
    print(f"Current (broken):  {current:+7.2f} MB")
    print(f"Fix #1 (new dict): {fix1:+7.2f} MB")
    print(f"Fix #2 (explicit): {fix2:+7.2f} MB")
    print(f"Fix #3 (rebuild):  {fix3:+7.2f} MB")
    
    print("\n" + "=" * 70)
    if fix1 < -10 and fix1 < current:
        print("✓ FIX #1 WORKS: Creating new state dict fixes the leak!")
        print("  Recommendation: Use .clone() when slicing optimizer state")
    elif fix2 < -10 and fix2 < current:
        print("✓ FIX #2 WORKS: Explicit deletion fixes the leak!")
        print("  Recommendation: Delete old tensors before reassignment")
    elif fix3 < -10 and fix3 < current:
        print("✓ FIX #3 WORKS: Rebuilding optimizer state fixes the leak!")
        print("  Recommendation: Clear state and let optimizer reinitialize")
        print("  Warning: Loses momentum (Adam state)")
    else:
        print("⚠️ None of the fixes worked - need deeper investigation")
    print("=" * 70)
