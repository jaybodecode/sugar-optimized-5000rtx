# PLY Metadata - iteration_7000.ply

**Filename:** `point_cloud_metadata_7000.md` (saved next to `point_cloud.ply`)

---

## Model Information
- **Iteration:** 7000 / 30000 (23.3% complete)
- **Gaussian Count:** 4,923,456 points
- **Scene:** garden (from `../SAMPLES/garden`)
- **Scene Extent:** 12.45 meters

---

## Training Metrics (at save time)
- **Loss (EMA):** 0.0247
- **L1 Loss:** 0.0231
- **Total Loss:** 0.0289 (L1 + DSSIM weighted)

---

## Test Quality Metrics
*Only available if iteration in --test_iterations*

**Evaluated at iteration 7000:**
- **PSNR:** 28.48 dB
- **SSIM:** 0.9234
- **LPIPS:** 0.0432 *(if --enable_lpips True)*
- **L1 Loss:** 0.0247

---

## Training Configuration
- **Resolution Divisor:** -r 2 (half-resolution)
- **Image Size:** 1244×1920 px (training), 2488×3840 px (original)
- **Training Cameras:** 141 images
- **Test Cameras:** 20 images
- **SH Degree:** 3 (max spherical harmonics)
- **Kernel Size:** 0.1 (anti-aliasing)

---

## Optimization Parameters
- **Position LR:** 0.00016 → 0.0000016 (exponential decay)
- **Feature LR:** 0.0025
- **Opacity LR:** 0.05
- **Scaling LR:** 0.005
- **Rotation LR:** 0.001
- **λ DSSIM:** 0.2 (20% SSIM, 80% L1)

---

## Densification Status
- **Phase:** Densification active (iters 500-15000)
- **Densify From:** 500
- **Densify Until:** 15000
- **Densification Interval:** Every 100 iterations
- **Opacity Reset Interval:** Every 3000 iterations
- **Last Opacity Reset:** Iteration 6000

---

## Training Command
```bash
python train.py -s ../SAMPLES/garden --iteration 30000 --test_iterations 1000 2500 5000 7000 10000 15000 20000 25000 30000 --checkpoint_iterations 2500 7000 15000 22000 30000 --save_iterations 7000 15000 30000 -r 2 --eval_camera_stride 1 --experiment_name "garden-r2-30k"
```

---

## File Locations
- **PLY File:** `point_cloud/iteration_7000/point_cloud.ply`
- **Checkpoint:** `chkpnt7000.pth` *(if --checkpoint_iterations 7000)*
- **Output Path:** `../SAMPLES/garden_output/garden-r2-30k`

---

**Generated:** 2026-01-27 15:35:42  
**File Size:** ~1.24 GB (4.92M Gaussians × ~260 bytes/point)

**Note:** Focuses on PLY model quality (PSNR, SSIM, LPIPS) and configuration for reproducibility.

---

**Quick Quality Check:**
- ✅ PSNR > 28 dB = Good quality for this scene
- ✅ Loss < 0.03 = Converging well
- ✅ 4.9M Gaussians = Typical for Garden scene
- ✅ VRAM < 16 GB = Stable, no spillover
