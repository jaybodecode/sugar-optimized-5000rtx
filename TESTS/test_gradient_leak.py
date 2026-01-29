#!/usr/bin/env python3
"""
Micro-test: Check if viewspace_point_tensor.grad accumulation causes memory leak
"""
import torch
import gc

def test_grad_accumulation_with_detach():
    """Test if .detach() prevents memory leak in gradient accumulation"""
    print("Test 1: Gradient accumulation WITH .detach()")
    
    torch.cuda.empty_cache()
    gc.collect()
    
    # Simulate densification stats accumulation
    n_gaussians = 500000  # 500K Gaussians
    accum = torch.zeros((n_gaussians, 1), device="cuda")
    
    initial_mem = torch.cuda.memory_allocated() / (1024**2)  # MB
    print(f"Initial VRAM: {initial_mem:.2f} MB")
    
    # Simulate 1000 iterations
    for i in range(1000):
        # Create viewspace_point_tensor with grad (simulates render output)
        viewspace = torch.randn((n_gaussians, 4), device="cuda", requires_grad=True)
        
        # Simulate backward pass
        loss = viewspace.sum()
        loss.backward()
        
        # Accumulate gradients WITH .detach() (like we'd fix it)
        update_filter = torch.rand(n_gaussians, device="cuda") > 0.5
        accum[update_filter] += torch.norm(viewspace.grad[update_filter, :2].detach(), dim=-1, keepdim=True)
        
        # Cleanup
        del viewspace, loss
        
        if i % 100 == 0:
            mem = torch.cuda.memory_allocated() / (1024**2)
            print(f"Iter {i:4d}: {mem:.2f} MB (delta: {mem-initial_mem:.2f} MB)")
    
    final_mem = torch.cuda.memory_allocated() / (1024**2)
    growth = final_mem - initial_mem
    print(f"Final VRAM: {final_mem:.2f} MB (growth: {growth:.2f} MB)")
    
    del accum
    torch.cuda.empty_cache()
    return growth

def test_grad_accumulation_without_detach():
    """Test if missing .detach() causes memory leak"""
    print("\nTest 2: Gradient accumulation WITHOUT .detach()")
    
    torch.cuda.empty_cache()
    gc.collect()
    
    n_gaussians = 500000
    accum = torch.zeros((n_gaussians, 1), device="cuda")
    
    initial_mem = torch.cuda.memory_allocated() / (1024**2)
    print(f"Initial VRAM: {initial_mem:.2f} MB")
    
    for i in range(1000):
        viewspace = torch.randn((n_gaussians, 4), device="cuda", requires_grad=True)
        loss = viewspace.sum()
        loss.backward()
        
        # Accumulate WITHOUT .detach() (current code)
        update_filter = torch.rand(n_gaussians, device="cuda") > 0.5
        accum[update_filter] += torch.norm(viewspace.grad[update_filter, :2], dim=-1, keepdim=True)
        
        del viewspace, loss
        
        if i % 100 == 0:
            mem = torch.cuda.memory_allocated() / (1024**2)
            print(f"Iter {i:4d}: {mem:.2f} MB (delta: {mem-initial_mem:.2f} MB)")
    
    final_mem = torch.cuda.memory_allocated() / (1024**2)
    growth = final_mem - initial_mem
    print(f"Final VRAM: {final_mem:.2f} MB (growth: {growth:.2f} MB)")
    
    del accum
    torch.cuda.empty_cache()
    return growth

if __name__ == "__main__":
    print("=" * 60)
    print("Memory Leak Test: Gradient Accumulation")
    print("=" * 60)
    
    growth_with = test_grad_accumulation_with_detach()
    growth_without = test_grad_accumulation_without_detach()
    
    print("\n" + "=" * 60)
    print("RESULTS:")
    print(f"  WITH .detach():    {growth_with:.2f} MB growth")
    print(f"  WITHOUT .detach(): {growth_without:.2f} MB growth")
    print(f"  Difference:        {abs(growth_without - growth_with):.2f} MB")
    
    if abs(growth_without - growth_with) > 10:  # >10MB difference
        print("\n⚠️  LEAK CONFIRMED: Missing .detach() causes memory leak!")
    else:
        print("\n✓  NO LEAK: .detach() makes no significant difference")
    print("=" * 60)
