# Training Speed Tracking

## Overview
Tracking iteration speed (it/s) improvements across optimization phases.

## Benchmark Configuration
- Model: mip-splatting
- Dataset: SAMPLES/garden
- Resolution: -r4 (quarter resolution)
- Training: 1000 iterations from baseline checkpoint (iter 3000→4000)
- Test Evaluation: Iteration 3100 (with --eval flag: 141 train, 20 test images)
- Hardware: RTX 5060 Ti 16GB GDDR7
- Environment: rtx5000_fresh (Python 3.11.14)

## Quality Metrics Baseline
**All optimizations must maintain these quality levels (±0.5% tolerance):**
- **PSNR:** ~26.0 dB
- **SSIM:** ~0.789
- **LPIPS:** ~0.241
- **L1 Loss:** ~0.0354

*These metrics are measured at iteration 3100 on the held-out test set. Any significant degradation indicates optimization introduced quality loss.*

---

## Results

### Baseline
- **Speed:** 11.06 it/s
- **Loss:** 0.0576
- **Peak VRAM (training):** 8.1 GB
- **Peak VRAM (eval):** ~16.0 GB (estimated)
- Date: Jan 27, 2026
- Notes: Reference implementation, no optimizations

### Phase 1 - Register Spilling Optimization (BASELINE FOR PHASE 2/3)
- **Speed:** 11.12 it/s (+0.06 it/s, +0.5%)
- **Loss:** 0.0578
- **Peak VRAM (training):** 7.8 GB (-300 MB)
- **Peak VRAM (eval):** 15.62 GB (-380 MB from baseline)
- **Shared GPU Memory (eval):** Visible spike in Windows Task Manager during eval phase
- Date: Jan 27, 2026
- Code Changes:
  - Added `__launch_bounds__(256)` to forward.cu (line ~165)
  - Added `__launch_bounds__(256)` to backward.cu (line ~381)
- Notes: Minimal speed impact, small VRAM reduction. Eval spike still at 15.6 GB causes GPU shared memory usage (screenshot evidence). Phase 2/3 needed to reduce eval peak below 16 GB.
- **Monitoring Setup:** VmSize tracking (monitor_vram_v2.sh) + manual Windows Task Manager screenshots for shared GPU memory

### Phase 2d - torch.no_grad() Evaluation Fix ✅ SUCCESS!
- **Speed:** 11.36 it/s (+0.24 it/s from Phase 1, +2.2%)
- **Loss:** 0.0577
- **Peak VRAM (training):** 8.0 GB (same as Phase 1)
- **Peak VRAM (eval):** 8.0 GB (**-7.62 GB from Phase 1!** No eval spike!)
- **Shared GPU Memory (eval):** **ZERO** - Completely eliminated spillover!
- Date: Jan 27, 2026
- Code Changes:
  - Added `with torch.no_grad():` wrapper around metric computation (train.py line ~772)
  - Fixed lpips_fn scoping issue (moved del outside for loop)
  - Prevents autograd graph from keeping all test images in VRAM
- Notes: **MAJOR SUCCESS!** Root cause was missing torch.no_grad() in evaluation loop. Without it, PyTorch built autograd graph keeping all 23 GT images + rendered images in VRAM. With no_grad(), only 1 image pair on GPU at a time. Eval peak dropped from 15.62 GB to 8.0 GB (51% reduction). Zero shared GPU memory confirmed in Windows Task Manager screenshot.
- **Attempts before success:** Phase 2/2b/2c (CPU GT caching, TensorBoard on CPU) had no effect - all showed 15.6 GB. Phase 2d torch.no_grad() was the actual fix.

### Phase 2d-eval2 - With --eval Flag ✅ VERIFIED!
- **Speed:** 11.42 it/s (+0.30 it/s from Phase 1, +2.7%)
- **Loss:** 0.0558
- **Peak VRAM (training):** 8.1 GB
- **Peak VRAM (eval):** **4.7 GB** (even lower with held-out test set!)
- **Test Metrics (Real generalization @ iter 3100):**
  - Test L1: 0.035384
  - Test PSNR: 26.01 dB ✅
  - Test SSIM: 0.7890 ✅
  - Test LPIPS: 0.2412 ✅
- Date: Jan 27, 2026
- Configuration: `--eval` flag enabled (141 train, 20 test images)
- Notes: With proper train/test split, eval VRAM even LOWER (4.7 GB vs 8.0 GB) because fewer test cameras (20 vs 185). Confirms torch.no_grad() fix works perfectly with real held-out test data. LPIPS working correctly after scope fix. **BASELINE QUALITY METRICS ESTABLISHED.**

### pynvml GPU Monitoring Optimization ❌ NO SPEEDUP
- **Speed:** 11.40 it/s (-0.02 it/s from Phase 2d-eval2, -0.2%)
- **Loss:** 0.0559
- **Peak VRAM (training):** 8.0 GB (50% usage)
- **Peak VRAM (eval):** 4.7 GB
- **Test Metrics (Quality verification @ iter 3100):**
  - Test L1: 0.035384 (✅ same)
  - Test PSNR: 26.01 dB (✅ same)
  - Test SSIM: 0.7890 (✅ same)
  - Test LPIPS: 0.2412 (✅ same)
- Date: Jan 27, 2026
- Code Changes:
  - Replaced nvidia-smi subprocess calls with pynvml direct API
  - Increased monitoring frequency (every 50 iterations vs 1000)
  - Added nvml_handle parameter threading through enhanced_training_report
- Notes: Expected 5-10% speedup from eliminating subprocess overhead, but actual result shows NO improvement. Possible reasons: (1) nvidia-smi overhead already negligible at 1000-iter intervals, (2) increased monitoring frequency (50 iter) may have offset gains, (3) pynvml overhead comparable to subprocess in this use case. **CONCLUSION:** Not worth the added complexity. Consider reverting unless monitoring frequency is critical.
- **Quality Impact:** NONE - All metrics identical to baseline

### Aggressive Pruning Test - Full 30K Run at -r2 ⚠️ VRAM DATA QUESTIONABLE
- **Speed:** 6.18 it/s
- **Loss:** 0.0501 (final training loss)
- **Peak VRAM (reported):** 1.87 GB ⚠️ **LIKELY INCORRECT - needs verification**
- **Peak VRAM (actual):** Unknown - monitoring may have failed
- **Test Metrics (Real generalization @ iter 30000):**
  - Test L1: 0.030744
  - Test PSNR: 26.24 dB
  - Test SSIM: 0.8050
  - Test LPIPS: 0.2007
- **Train Metrics @ iter 30000:**
  - Train L1: 0.025119
  - Train PSNR: 28.12 dB
  - Train SSIM: 0.8489
  - Train LPIPS: 0.1913
- **Model Size:** 1,811,372 Gaussians (1811.4K)
- **Average Opacity:** 0.749
- Date: Jan 27, 2026
- Configuration: 
  - Resolution: -r2 (half resolution)
  - Started from checkpoint at iter 7000
  - Ran to iter 30000 (23K iterations)
  - Experiment: garden-aggressive-pruning-v1
- Notes: Peak VRAM reading of 1.87 GB is suspiciously low for -r2 full training. Possible issues: (1) VRAM monitoring only captured post-training cleanup state, (2) monitoring interval missed actual peaks, (3) checkpoint resume may have reset monitoring. **NEEDS RE-RUN WITH PROPER VRAM TRACKING.** Quality metrics show good generalization (test PSNR 26.24 dB).

### Phase 3 - Clear Test Cache (Pending - May Not Be Needed)
- **Status:** On hold - Phase 2d already eliminated eval spike
- **Current eval VRAM:** 8.0 GB (well below 16 GB limit)
- **Shared GPU Memory:** Already zero
- Potential Changes:
  - Clear cached test data after evaluation in train.py
  - Add explicit cache clearing between test/train evaluation sets
- **Decision:** Only implement if higher resolution (-r2, -r1) shows VRAM issues

---

## Comparison Table

| Phase | Speed (it/s) | Change | Train VRAM | Eval VRAM | PSNR (dB) | SSIM | LPIPS | Quality | Notes |
|-------|--------------|---------|------------|-----------|-----------|------|-------|---------|-------|
| Baseline | 11.06 | - | 8.1 GB | ~16.0 GB | - | - | - | - | -r4, No --eval |
| Phase 1 | 11.12 | +0.5% | 7.8 GB | 15.6 GB | - | - | - | - | -r4, No --eval |
| Phase 2d | 11.36 | +2.7% | 8.0 GB | **8.0 GB** | - | - | - | - | -r4, No --eval |
| Phase 2d-eval2 | 11.42 | +3.3% | 8.1 GB | **4.7 GB** | 26.01 | 0.7890 | 0.2412 | ✅ Baseline | -r4, With --eval |
| pynvml | 11.40 | +3.1% | 8.0 GB | 4.7 GB | 26.01 | 0.7890 | 0.2412 | ✅ Same | -r4, No speedup |
| Aggressive Prune | 6.18 | - | ⚠️ 1.87 GB? | ? | 26.24 | 0.8050 | 0.2007 | ✅ Good | -r2, 30K, VRAM data suspect |
| Phase 3 | On hold | - | - | - | - | - | - | - | Not needed |

---

## Measurement Notes
- **VRAM:** Measured via nvidia-smi and monitor_vram_v2.sh
- **Shared GPU Memory:** Manual observation via Windows Task Manager (WSL2 limitation)
- **VmSize tracking:** Attempted but doesn't capture CUDA driver-level allocations in WSL2
- **Key metric for spillover:** Presence of "Shared GPU memory" usage in Windows Task Manager
- **Goal:** Keep peak VRAM below 16 GB hardware limit to eliminate spillover entirely
- **Phase 2d Achievement:** Eval VRAM reduced 51% (15.62 GB → 8.0 GB), zero spillover confirmed
