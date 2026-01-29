#!/usr/bin/env python3
"""
Micro-test: Check if optimizer state (Adam) is properly freed during pruning
This is the most likely leak source since Adam stores 2× parameter memory.
"""
import torch
import torch.nn as nn
import gc

class GaussianModel(nn.Module):
    """Simplified Gaussian model for testing"""
    def __init__(self, n_gaussians):
        super().__init__()
        self.xyz = nn.Parameter(torch.randn((n_gaussians, 3), device="cuda"))
        self.opacity = nn.Parameter(torch.randn((n_gaussians, 1), device="cuda"))
        self.scaling = nn.Parameter(torch.randn((n_gaussians, 3), device="cuda"))
        self.rotation = nn.Parameter(torch.randn((n_gaussians, 4), device="cuda"))
        self.features = nn.Parameter(torch.randn((n_gaussians, 48), device="cuda"))
        
    def forward(self):
        return (self.xyz.sum() + self.opacity.sum() + self.scaling.sum() + 
                self.rotation.sum() + self.features.sum())

def get_optimizer_state_size(optimizer):
    """Calculate total memory used by optimizer state"""
    total_bytes = 0
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state.get(p, None)
            if state is not None:
                # Adam stores exp_avg and exp_avg_sq
                if 'exp_avg' in state:
                    total_bytes += state['exp_avg'].element_size() * state['exp_avg'].nelement()
                if 'exp_avg_sq' in state:
                    total_bytes += state['exp_avg_sq'].element_size() * state['exp_avg_sq'].nelement()
    return total_bytes / (1024**2)  # Convert to MB

def test_optimizer_state_with_pruning():
    """Test if optimizer state is freed when parameters are pruned"""
    print("Test: Optimizer State Memory During Pruning")
    print("=" * 70)
    
    torch.cuda.empty_cache()
    gc.collect()
    
    # Create model with 500K Gaussians
    n_gaussians = 500000
    model = GaussianModel(n_gaussians)
    
    # Create Adam optimizer (like the real training)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Run a few iterations to initialize optimizer state (exp_avg, exp_avg_sq)
    print("Initializing optimizer state (3 iterations)...")
    for i in range(3):
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()
    
    initial_mem = torch.cuda.memory_allocated() / (1024**2)
    initial_opt_mem = get_optimizer_state_size(optimizer)
    n_params = sum(p.numel() for p in model.parameters())
    
    print(f"\nInitial state (500K Gaussians):")
    print(f"  Total VRAM: {initial_mem:.2f} MB")
    print(f"  Parameters: {n_params:,} ({n_params*4/(1024**2):.2f} MB)")
    print(f"  Optimizer state: {initial_opt_mem:.2f} MB")
    print(f"  Ratio: {initial_opt_mem / (n_params*4/(1024**2)):.2f}× (should be ~2× for Adam)")
    
    # Simulate pruning: Remove 37% (185K Gaussians)
    prune_ratio = 0.37
    n_to_prune = int(n_gaussians * prune_ratio)
    valid_indices = torch.arange(n_gaussians, device="cuda")[n_to_prune:]  # Keep last 63%
    
    print(f"\nPruning {n_to_prune:,} Gaussians ({prune_ratio*100:.1f}%)...")
    
    # Simulate _prune_optimizer (WITH FIX from gaussian_model.py)
    for group in optimizer.param_groups:
        for i, p in enumerate(group['params']):
            state = optimizer.state.get(p, None)
            if state is not None:
                # FIX: Create new sliced tensors
                new_exp_avg = state["exp_avg"][valid_indices]
                new_exp_avg_sq = state["exp_avg_sq"][valid_indices]
                
                # Delete old tensors explicitly
                del state["exp_avg"]
                del state["exp_avg_sq"]
                
                # Assign new sliced tensors
                state["exp_avg"] = new_exp_avg
                state["exp_avg_sq"] = new_exp_avg_sq
                
                # Delete old optimizer state entry
                del optimizer.state[p]
                
                # Create new parameter (sliced)
                new_param = nn.Parameter(p.data[valid_indices].requires_grad_(True))
                group['params'][i] = new_param
                
                # Assign state to new parameter
                optimizer.state[new_param] = state
                
                # Update model parameter
                if i == 0:
                    model.xyz = new_param
                elif i == 1:
                    model.opacity = new_param
                elif i == 2:
                    model.scaling = new_param
                elif i == 3:
                    model.rotation = new_param
                elif i == 4:
                    model.features = new_param
    
    gc.collect()
    torch.cuda.empty_cache()
    
    after_prune_mem = torch.cuda.memory_allocated() / (1024**2)
    after_prune_opt_mem = get_optimizer_state_size(optimizer)
    n_params_after = sum(p.numel() for p in model.parameters())
    
    print(f"\nAfter pruning:")
    print(f"  Total VRAM: {after_prune_mem:.2f} MB")
    print(f"  Parameters: {n_params_after:,} ({n_params_after*4/(1024**2):.2f} MB)")
    print(f"  Optimizer state: {after_prune_opt_mem:.2f} MB")
    
    memory_freed = initial_mem - after_prune_mem
    opt_state_freed = initial_opt_mem - after_prune_opt_mem
    param_freed = (n_params - n_params_after) * 4 / (1024**2)
    
    print(f"\nMemory changes:")
    print(f"  Total freed: {memory_freed:.2f} MB")
    print(f"  Params freed: {param_freed:.2f} MB")
    print(f"  Opt state freed: {opt_state_freed:.2f} MB")
    
    expected_param_freed = n_to_prune * (3 + 1 + 3 + 4 + 48) * 4 / (1024**2)
    expected_opt_freed = expected_param_freed * 2  # Adam stores 2× (exp_avg + exp_avg_sq)
    expected_total_freed = expected_param_freed + expected_opt_freed
    
    print(f"\nExpected freed:")
    print(f"  Params: {expected_param_freed:.2f} MB")
    print(f"  Opt state: {expected_opt_freed:.2f} MB")
    print(f"  Total: {expected_total_freed:.2f} MB")
    
    print(f"\nEfficiency:")
    print(f"  Params: {param_freed/expected_param_freed*100:.1f}%")
    print(f"  Opt state: {opt_state_freed/expected_opt_freed*100:.1f}%")
    print(f"  Total: {memory_freed/expected_total_freed*100:.1f}%")
    
    return memory_freed, expected_total_freed

def test_multiple_prune_cycles():
    """Test if memory leaks accumulate over multiple prune cycles"""
    print("\n" + "=" * 70)
    print("Test: Multiple Prune Cycles (checking for accumulation)")
    print("=" * 70)
    
    torch.cuda.empty_cache()
    gc.collect()
    
    n_gaussians = 300000
    model = GaussianModel(n_gaussians)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Initialize optimizer
    for _ in range(3):
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()
    
    initial_mem = torch.cuda.memory_allocated() / (1024**2)
    print(f"Initial VRAM: {initial_mem:.2f} MB")
    
    # Perform 5 prune cycles
    for cycle in range(1, 6):
        n_current = sum(p.numel() // 59 for p in model.parameters())  # 59 = 3+1+3+4+48
        n_to_prune = int(n_current * 0.1)  # Prune 10% each time
        
        print(f"\nCycle {cycle}: Pruning {n_to_prune:,} Gaussians (10%)")
        
        # Prune (with FIX)
        keep_indices = torch.arange(n_current - n_to_prune, device="cuda")
        
        for group in optimizer.param_groups:
            for i, p in enumerate(group['params']):
                state = optimizer.state.get(p, None)
                if state is not None:
                    # FIX: Create new, delete old
                    new_exp_avg = state["exp_avg"][keep_indices]
                    new_exp_avg_sq = state["exp_avg_sq"][keep_indices]
                    del state["exp_avg"]
                    del state["exp_avg_sq"]
                    state["exp_avg"] = new_exp_avg
                    state["exp_avg_sq"] = new_exp_avg_sq
                    
                    del optimizer.state[p]
                    new_param = nn.Parameter(p.data[keep_indices].requires_grad_(True))
                    group['params'][i] = new_param
                    optimizer.state[new_param] = state
        
        torch.cuda.empty_cache()
        gc.collect()
        
        current_mem = torch.cuda.memory_allocated() / (1024**2)
        print(f"  VRAM: {current_mem:.2f} MB (change: {current_mem - initial_mem:+.2f} MB)")
    
    final_mem = torch.cuda.memory_allocated() / (1024**2)
    total_change = final_mem - initial_mem
    
    print(f"\nFinal VRAM: {final_mem:.2f} MB")
    print(f"Total change: {total_change:+.2f} MB")
    
    if total_change > 5:  # More than 5MB growth after pruning 40%
        print("⚠️  LEAK DETECTED: Memory grew despite pruning!")
    else:
        print("✓  No leak: Memory decreased as expected")

if __name__ == "__main__":
    freed, expected = test_optimizer_state_with_pruning()
    test_multiple_prune_cycles()
    
    print("\n" + "=" * 70)
    print("FINAL VERDICT:")
    print("=" * 70)
    
    if freed < expected * 0.7:  # Less than 70% freed
        print("⚠️  OPTIMIZER STATE LEAK CONFIRMED!")
        print(f"   Expected {expected:.2f} MB freed, only got {freed:.2f} MB")
        print("   Optimizer state is NOT being properly freed during pruning")
    else:
        print("✓  NO LEAK: Optimizer state is properly freed during pruning")
        print(f"   Freed {freed:.2f} MB / Expected {expected:.2f} MB ({freed/expected*100:.1f}%)")
    print("=" * 70)
