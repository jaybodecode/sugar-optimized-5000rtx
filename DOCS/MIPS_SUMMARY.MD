# Mip-Splatting VRAM Optimization Summary

## Problem Statement
Mip-splatting evaluation was causing VRAM to spike from 8 GB (training) to 16 GB (evaluation), causing GPU memory spillover to system RAM and significant slowdowns.

## Root Cause Analysis

### The Discovery Process
**Initial hypothesis (WRONG):** GT images were being cached on GPU
- Attempted Phase 2/2b/2c: Moved GT images to CPU, computed visualizations on CPU
- **Result:** No improvement - still 15.6 GB during eval

**Root cause (CORRECT):** Missing `torch.no_grad()` wrapper in evaluation loop

### Why It Mattered

**Without `torch.no_grad()` (Phase 1 - 15.6 GB eval):**
```python
for idx, viewpoint in enumerate(config['cameras']):
    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
    gt_image = viewpoint._cached_gt_cpu.cuda()
    
    # PyTorch builds autograd graph here!
    l1_test += l1_loss(image, gt_image).mean().double()
    psnr_test += psnr(image, gt_image).mean().double()
    # All 23 images accumulate in VRAM via autograd graph
```

**With `torch.no_grad()` (Phase 2d - 8.0 GB eval):**
```python
for idx, viewpoint in enumerate(config['cameras']):
    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
    gt_image = viewpoint._cached_gt_cpu.cuda()
    
    # No autograd graph - tensors released after each iteration
    with torch.no_grad():
        l1_test += l1_loss(image, gt_image).mean().double()
        psnr_test += psnr(image, gt_image).mean().double()
    
    del gt_image  # Actually freed now!
```

## Technical Explanation

### What is torch.no_grad()?

PyTorch by default tracks all tensor operations to build a **computation graph** for backpropagation (training). During evaluation, we don't need this graph, but PyTorch doesn't know that!

**Autograd Graph Retention:**
- Without `torch.no_grad()`: PyTorch keeps ALL intermediate tensors in memory
- Every `gt_image`, `image`, loss computation creates references
- Graph accumulates: Image 1 + Image 2 + ... + Image 23 = 15.6 GB
- Even explicit `del gt_image` doesn't free it - autograd holds references!

**With torch.no_grad():**
- PyTorch doesn't build computation graph
- Tensors freed immediately when deleted
- Only current image pair in VRAM at any time
- Memory usage: constant 8 GB regardless of test set size

### Why Training Uses 8 GB But Eval Used 15.6 GB?

**Training Loop (8 GB - CORRECT):**
```python
# This IS wrapped in training context, gradients cleared each iteration
loss.backward()  # Uses gradients
optimizer.step()
optimizer.zero_grad(set_to_none=True)  # Clears graph!
```

**Evaluation Loop (15.6 GB - BROKEN):**
```python
# No backward pass, but autograd STILL tracking!
# Graph accumulates across all test images
# Nothing clears it until loop completes
```

## The Solution

### Code Changes (train.py line ~772)
```python
# Phase 2d: Wrap metric computation in no_grad() to prevent autograd graph
with torch.no_grad():
    l1_test += l1_loss(image, gt_image).mean().double()
    psnr_test += psnr(image, gt_image).mean().double()
    ssim_test += ssim(image, gt_image).mean().double()
    if opt.enable_lpips:
        lpips_test += lpips_fn(image, gt_image).mean().double()
```

### Additional Fix (lpips_fn scoping)
Moved `del lpips_fn` outside the `for config` loop to prevent UnboundLocalError when evaluating multiple configs (test + train).

## Results

### Performance Impact
| Metric | Before (Phase 1) | After (Phase 2d) | Improvement |
|--------|------------------|------------------|-------------|
| Training VRAM | 7.8 GB | 8.0 GB | -200 MB (negligible) |
| Eval VRAM | 15.6 GB | 8.0 GB | **-7.6 GB (51%)** |
| Shared GPU Memory | Yes (spillover) | **ZERO** | Eliminated |
| Training Speed | 11.12 it/s | 11.36 it/s | +2.2% |

### With --eval Flag (Proper Test Split)
| Metric | Value |
|--------|-------|
| Training VRAM | 8.1 GB |
| Eval VRAM | **4.7 GB** (even lower!) |
| Test PSNR | 26.01 dB |
| Test SSIM | 0.7889 |

## Key Learnings

### 1. Training vs Evaluation Memory Usage
**Common misconception:** "Evaluation should use less memory because no backward pass"

**Reality:** 
- Training: Gradients cleared after each iteration (optimizer.zero_grad)
- Evaluation (without no_grad): Autograd graph accumulates indefinitely
- **Evaluation can use MORE memory than training if not careful!**

### 2. Why --eval Affects VRAM
Not because it disables gradients (that's `torch.no_grad()`), but because:
- Fewer test cameras (20 vs 185)
- Less data to accumulate in autograd graph
- Phase 2d with --eval: 4.7 GB (20 images)
- Phase 2d without --eval: 8.0 GB (185 images)

### 3. Debugging GPU Memory in WSL2
**Challenge:** CUDA shared memory (spillover) not visible in Linux tools
- `/proc/[pid]/status` VmSize - doesn't show CUDA allocations
- RSS (Resident Set) - only process RAM, not GPU
- nvidia-smi - shows VRAM but not spillover

**Solution:** Windows Task Manager
- "Shared GPU memory" chart shows actual spillover
- Only reliable metric for detecting swap to system RAM

## Recommendations

### For Other Projects
1. **Always use torch.no_grad() during evaluation/inference**
   ```python
   with torch.no_grad():
       for data in test_loader:
           output = model(data)
           # compute metrics
   ```

2. **Don't assume evaluation uses less memory than training**
   - Training clears gradients each iteration
   - Evaluation may accumulate without proper no_grad()

3. **Profile VRAM during evaluation phases**
   - Peak VRAM often occurs during eval, not training
   - Monitor with nvidia-smi or Windows Task Manager

### For This Project (Future Work)
- Phase 3 (cache clearing) not needed at -r4 resolution
- Test at higher resolutions (-r2, -r1) to see if further optimization needed
- Current solution handles 16 GB VRAM budget comfortably

## References
- Original issue: VRAM spike 8 GB → 16 GB during evaluation
- Benchmark: SAMPLES/garden dataset, -r4 resolution
- Hardware: RTX 5000 Ada (16 GB VRAM)
- Date: January 27, 2026
- Related docs: [SPEEDUP.md](../mip-splatting/SPEEDUP.md), [MIPS_OPTIMISATION.MD](MIPS_OPTIMISATION.MD)
