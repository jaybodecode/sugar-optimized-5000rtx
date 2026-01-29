# Mip-Splatting Analysis Tools

## analyze_ply.py - Gaussian Splatting PLY Analyzer

Standalone tool for comprehensive analysis of Gaussian Splatting PLY files.

### Features

**Model Statistics:**
- Gaussian point count
- File size and bytes per point
- Bounding box (X, Y, Z ranges)
- Scene center and extent (radius)
- Furthest point from center

**Gaussian Attributes:**
- Opacity distribution (detects invisible Gaussians)
- Scale magnitudes (detects artifacts/noise)
- 3D Filter values (Mip-Splatting custom attribute)
- Spherical harmonics (SH coefficients, base color)

**Quality Metrics (optional):**
- Iteration number
- Training loss
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)

**Defect Detection:**
- NaN/Inf values in positions
- Spatial outliers (>1.5x scene extent)
- Low opacity points (potential culling candidates)
- Huge scales (>10.0) - potential artifacts
- Tiny scales (<0.0001) - potential noise

### Usage

**Basic analysis (PLY file only):**
```bash
python analyze_ply.py point_cloud.ply
```

**With training metadata:**
```bash
python analyze_ply.py point_cloud.ply --iteration 7000 --loss 0.0247
```

**With quality metrics (PSNR, SSIM, LPIPS):**
```bash
python analyze_ply.py point_cloud.ply --psnr 28.5 --ssim 0.92 --lpips 0.043
```

**Full example (all metadata):**
```bash
python analyze_ply.py point_cloud.ply \
  --iteration 7000 \
  --loss 0.0247 \
  --psnr 28.48 \
  --ssim 0.9234 \
  --lpips 0.0432
```

### Integration with train.py

The analyzer is automatically called after each PLY save during training:
- Runs after `scene.save(iteration)` in train.py
- Includes current iteration, loss (EMA)
- Includes test metrics (PSNR, SSIM, LPIPS) if available from last evaluation

**Example output during training:**
```
Iter 7000: Saving Gaussians
Analyzing PLY file...

================================================================================
PLY ANALYSIS REPORT
================================================================================

📁 File: point_cloud.ply
📂 Path: .../point_cloud/iteration_7000
💾 Size: 1037.37 MB

────────────────────────────────────────────────────────────────────────────
MODEL STATISTICS
────────────────────────────────────────────────────────────────────────────

🎯 Gaussian Count: 4,386,142 points
📊 Bytes per Point: 248.0 bytes

────────────────────────────────────────────────────────────────────────────
SPATIAL BOUNDS
────────────────────────────────────────────────────────────────────────────

X: [-44.923, +57.476]  (range: 102.399)
Y: [-34.003, +9.195]  (range: 43.198)
Z: [-24.118, +35.225]  (range: 59.343)

📍 Center: (-0.400, +0.559, +1.592)
📏 Extent (radius): 60.379 meters

🔭 Furthest Point: (-42.324, -32.078, +30.276)
   Distance from center: 60.379 meters

────────────────────────────────────────────────────────────────────────────
QUALITY METRICS
────────────────────────────────────────────────────────────────────────────

Iteration: 7000
Training Loss: 0.024700
PSNR:  28.48 dB
SSIM:  0.9234
LPIPS: 0.0432

────────────────────────────────────────────────────────────────────────────
DEFECT ANALYSIS
────────────────────────────────────────────────────────────────────────────

⚠️  Potential Issues Detected:
  • Many invisible Gaussians: 3,593,310 points (81.92%) with opacity < 0.001

================================================================================
```

### Dependencies

```bash
pip install plyfile numpy
```

### Use Cases

1. **Post-training analysis:** Inspect saved PLY files anytime
2. **Quality assurance:** Detect potential defects (NaN, huge scales, outliers)
3. **Model comparison:** Compare Gaussian count, opacity, scales across iterations
4. **Research:** Extract statistics for papers (bounding box, extent, distributions)
5. **Production:** Validate exported models before downstream use (SuGaR, viewers)

### Notes

- Works with any Gaussian Splatting PLY (not just Mip-Splatting)
- Detects Mip-Splatting custom attributes (filter_3D) if present
- Quality metrics (PSNR/SSIM/LPIPS) are optional - useful when provided by training script
- Zero computational overhead - only reads PLY file, no rendering
