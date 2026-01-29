# SuGaR Training Pipeline - Internal Workflow Documentation

## Overview

This document describes the **internal training workflow** of SuGaR - what happens when you run `python train.py`. For command-line arguments and usage, see [SUGAR_USAGE.MD](SUGAR_USAGE.MD).

**Pipeline Stages:**
1. **Coarse Training** - Optimize Gaussian positions/scales with regularization (~30-60 min)
2. **Mesh Extraction** - Extract surface mesh from trained Gaussians (~5-10 min)
3. **Mesh Refinement** - Fine-tune mesh-aligned Gaussians (~20-45 min)
4. **Texture Export** - Generate UV-textured .obj file (~5 min)

**Total Time:** ~1-2 hours for high-quality mesh with 60K checkpoint

---

## Training Phases Explained

### Phase 1: Coarse Training

**Purpose:** Optimize Gaussian positions, scales, and orientations to align with scene surface while maintaining rendering quality.

**Regularization Methods:**
- **`sdf`** - Signed Distance Field regularization (experimental)
- **`density`** - Density estimation regularization
- **`dn_consistency`** - Depth-Normal consistency (recommended, best quality)

**Key Operations:**
```python
# Location: SuGaR/sugar_trainers/coarse_density_and_dn_consistency.py

# 1. Load checkpoint
nerfmodel = GaussianSplattingWrapper(
    source_path=source_path,
    output_path=gs_checkpoint_path,
    iteration_to_load=iteration_to_load,  # e.g., 40000, 60000
)

# 2. Initialize SuGaR model with Training KNN (Stage 1)
sugar = SuGaR(
    nerfmodel=nerfmodel,
    points=points,
    colors=colors,
    keep_track_of_knn=True,  # Computes K=16 KNN for training regularization
    knn_to_track=16,
)

# 3. Training loop (7K-60K iterations)
for iteration in range(start_iteration, num_iterations):
    # Render image
    render_pkg = render(...)
    
    # Compute loss with regularization
    loss = (
        l1_loss(render_pkg['render'], gt_image) +
        ssim_loss(render_pkg['render'], gt_image) +
        density_regularization +
        normal_consistency_loss
    )
    
    # Backprop and optimize
    loss.backward()
    optimizer.step()
    
    # Save checkpoints at milestones
    if iteration in checkpoint_milestones:
        torch.save(sugar.state_dict(), f'{iteration}.pt')
```

**Training KNN (Stage 1):**
- **When:** Model initialization (beginning of Phase 1)
- **Purpose:** Track nearest neighbors for training regularization
- **K=16** neighbors for each Gaussian
- **Time:** ~3-5 minutes for 6M points
- **VRAM:** ~2-3 GB temporary spike
- **Cached:** Yes - saved in checkpoint, reused if resuming training
- **Output:** "✓ Using cached initialization... Skipping expensive KNN K=16 computation"

**Training Outputs:**
```
📁 Output Structure:
checkpoint_path_mesh/
└── experiment_name/
    └── sugarcoarse_3Dgs40000_densityestim02_sdfnorm02/
        ├── 61000.pt          # Checkpoint at 61K iterations
        ├── 63000.pt          # Checkpoint at 63K iterations
        ├── 65000.pt          # Final checkpoint
        └── tensorboard/      # Training logs
            └── events.out.tfevents...
```

**Iteration Resume Bug (FIXED):**
```python
# OLD BUG (before fix):
iteration = 7000 - 1  # Hardcoded! Ignored checkpoint iteration

# NEW (after fix):
iteration = iteration_to_load  # Properly resumes from checkpoint
```

**Why This Matters:**
- **Before fix:** Loading 40K checkpoint started training from 7K (rewrote 33K iterations!)
- **After fix:** Loading 40K checkpoint starts from 40K (proper fine-tuning)

---

### Phase 2: Mesh Extraction

**Purpose:** Convert trained Gaussian point cloud into triangulated surface mesh.

**Pipeline Overview:**
1. Load trained SuGaR checkpoint
2. Prune low-opacity Gaussians (< 0.5)
3. Extract 10M surface points from multiple camera views
4. Split foreground/background regions
5. Poisson surface reconstruction (depth=10)
6. Mesh decimation (200K or 1M vertices)
7. Optional: Project mesh onto surface points for detail preservation
8. Save as .ply file with vertex colors and normals

**Time:** ~5-10 minutes  
**VRAM:** ~8-10 GB peak (with `mesh_extraction_resolution_factor=0.5`)

**Key Bottlenecks:**
- Surface point extraction: 3-5 min (50-60% of Phase 2 time)
- Poisson reconstruction: 2-3 min (30-40%, CPU-bound)
- Mesh decimation: 2-3 min (20-30%)

**Mesh Extraction Outputs:**
```
📁 Output Structure:
checkpoint_path_mesh/
└── experiment_name/
    ├── sugarcoarse_3Dgs40000_densityestim02_sdfnorm02/
    │   └── (training checkpoints)
    └── mesh/
        ├── coarse_mesh_level0.3_decim1000000.ply  # Extracted mesh (1M vertices)
        └── coarse_mesh_level0.3_decim1000000_proj.ply  # Projected mesh (if enabled)
```

**📖 For Complete Stage 2 Analysis:**  
See **[SUGAR_TRAIN_MESH.MD](SUGAR_TRAIN_MESH.MD)** for comprehensive documentation including:
- All 10 mesh extraction steps with code locations
- VRAM usage and optimization strategies  
- Parameter impact tables (quality vs speed vs VRAM)
- Performance bottleneck analysis
- Known issues and fixes
- Recommended workflows for 16GB and 24GB+ VRAM

---

### Phase 3: Mesh Refinement

**Purpose:** Fine-tune Gaussians bound to mesh triangles for optimal rendering quality.

**Key Operations:**
```python
# Location: SuGaR/sugar_trainers/refine.py

# 1. Load coarse mesh
mesh = load_mesh(coarse_mesh_path)

# 2. Create refined SuGaR with Gaussians bound to mesh
refined_sugar = SuGaR(
    nerfmodel=nerfmodel,
    surface_mesh=mesh,
    n_gaussians_per_surface_triangle=1,  # High poly: 1 Gaussian/triangle
)

# 3. Training loop (2K-15K iterations)
for iteration in range(refinement_iterations):
    # Render with mesh-aligned Gaussians
    render_pkg = render(...)
    
    # Compute loss (image quality + normal consistency)
    loss = (
        l1_loss(render_pkg['render'], gt_image) +
        ssim_loss(render_pkg['render'], gt_image) +
        normal_consistency_loss * 0.1
    )
    
    # Optimize only Gaussian parameters (positions fixed to mesh)
    loss.backward()
    optimizer.step()
```

**Refinement Time Settings:**
- **`short`** - 2,000 iterations (~10-15 min) - Quick test
- **`medium`** - 7,000 iterations (~20-30 min) - Good balance
- **`long`** - 15,000 iterations (~40-60 min) - Best quality (recommended)

**Refinement Outputs:**
```
📁 Output Structure:
checkpoint_path_mesh/
└── experiment_name/
    ├── (coarse training + mesh)
    └── sugarfine_3Dgs40000_sdfnorm02_eval_mesh1000000vtx_gpt1/
        ├── refined_model.pt
        ├── refined_mesh.ply  (if --export_ply True)
        └── tensorboard/
```

---

### Phase 4: Texture Export

**Purpose:** Generate traditional UV-textured mesh (.obj + .mtl + .png) for use in standard 3D software.

**Key Operations:**
```python
# Location: SuGaR/sugar_extractors/refined_mesh.py

# 1. Load refined SuGaR
refined_sugar = SuGaR.load(refined_model_path)

# 2. Parameterize mesh with UV coordinates
uv_coords = compute_uv_parameterization(refined_sugar.mesh)

# 3. Render texture map from multiple views
texture_map = render_texture_from_gaussians(
    refined_sugar,
    uv_coords,
    square_size=8,  # Texture resolution
)

# 4. Export .obj with texture
export_obj(
    mesh=refined_sugar.mesh,
    uv_coords=uv_coords,
    texture_map=texture_map,
    output_path='textured_mesh.obj'
)
```

**Texture Outputs:**
```
📁 Output Structure:
checkpoint_path_mesh/
└── experiment_name/
    ├── (all previous outputs)
    └── textured_mesh/
        ├── mesh.obj           # Mesh with UV coordinates
        ├── mesh.mtl           # Material definition
        └── texture.png        # RGB texture map (8×8 or 16×16 squares)
```

---

## Progress Monitoring

### Live Progress Bar (Phase 1 & 3)

**Current Behavior:**
```
Training [=====>    ] 72% 46600/65000 │ L:0.373 │ V:82%/13.1GB │ R:33%/25.2GB │ 27.08it/s │ 4m │ ETA:11m
         ↑           ↑  ↑               ↑         ↑              ↑              ↑          ↑    ↑
         Bar         %  Iteration       Loss      VRAM           RAM            Speed      Time ETA
```

**Update Frequency:** Once per batch (~141 images in Garden dataset)
- Updates every ~5-10 seconds (slower than ideal)
- See [REFACTOR_SUGAR_PROGRESS_BAR.md](../NOGIT/REFACTOR_SUGAR_PROGRESS_BAR.md) for planned improvements

**Metrics:**
- **L (Loss):** Combined rendering + regularization loss (lower = better)
- **V (VRAM):** GPU memory usage (% / GB)
- **R (RAM):** System memory usage (% / GB)
- **it/s:** Iterations per second (depends on image count, ~20-30 typical)
- **ETA:** Estimated time remaining

### TensorBoard Logs

**Location:** `checkpoint_path_mesh/experiment_name/sugarcoarse_*/tensorboard/`

**Metrics Logged:**
- Render loss (L1 + SSIM)
- Density regularization loss
- Normal consistency loss
- Total combined loss
- PSNR (Peak Signal-to-Noise Ratio)
- Training speed (it/s)

**View Logs:**
```bash
# Compare all experiments for a checkpoint
tensorboard --logdir "../SAMPLES/garden_output/garden-r2-60k-6M-quality_mesh" --port 6007

# View single experiment
tensorboard --logdir "../SAMPLES/garden_output/garden-r2-60k-6M-quality_mesh/6M-quality-5Mverts/sugarcoarse_3Dgs40000_densityestim02_sdfnorm02/tensorboard" --port 6007
```

---

## Checkpointing System

### Coarse Training Checkpoints

**Checkpoint Types:**
1. **Milestone Checkpoints** - Specific iterations (e.g., 61000, 63000, 65000)
2. **Interval Checkpoints** - Every N iterations (e.g., every 1000)
3. **Final Checkpoint** - Last iteration (always saved)

**Configuration:**
```bash
--checkpoint_milestones 61000 63000 65000  # Save at these specific iterations
--checkpoint_interval 1000                  # Save every 1000 iterations
```

**Checkpoint Contents:**
- Model state (`sugar.state_dict()`)
- Optimizer state
- Training iteration
- Loss history
- KNN data (cached for faster resume)

**Resume Training:**
```bash
--resume_checkpoint path/to/checkpoint/65000.pt
```

---

## Memory Management

### VRAM Usage by Phase

**Phase 1 - Coarse Training:**
- Baseline: ~8-12 GB (depends on checkpoint size)
- Training KNN spike: +2-3 GB (temporary, ~3-5 min)
- Peak during training: ~12-16 GB

**Phase 2 - Mesh Extraction:**
- Mesh Extraction KNN spike: +2-3 GB (temporary, ~2-4 min)
- Mesh generation: ~4-6 GB
- Peak: ~8-10 GB

**Phase 3 - Mesh Refinement:**
- Baseline: ~6-10 GB (depends on mesh size)
- Peak during training: ~10-14 GB

**Phase 4 - Texture Export:**
- Texture rendering: ~4-6 GB
- Peak: ~6-8 GB

### VRAM Optimizations

**Enabled by Default:**
- Gradient checkpointing (saves 30-40% VRAM)
- Half-resolution depth-normal maps (saves ~1-2 GB)
- Lazy loading (caches computed values)
- Reduced SDF samples (1M → 250K)

**See Also:** [SUGAR_OPTIMISATIONS.MD](SUGAR_OPTIMISATIONS.MD)

---

## Performance Expectations

### Garden Dataset (161 images, 1244×1920)

**Using 60K Checkpoint (6M Gaussians):**

| Phase | Iterations | Time | VRAM Peak | Output |
|-------|-----------|------|-----------|--------|
| Coarse Training | 5K (60K→65K) | ~30 min | 15.7 GB | Checkpoints |
| Mesh Extraction | - | ~5 min | 10 GB | .ply mesh |
| Mesh Refinement | 15K (long) | ~45 min | 14 GB | Refined mesh |
| Texture Export | - | ~5 min | 8 GB | .obj + texture |
| **Total** | - | **~1.5 hrs** | **15.7 GB** | Complete pipeline |

**Speed Factors:**
- Image count (more images = slower)
- Image resolution (higher = slower)
- Checkpoint size (more Gaussians = slower)
- GPU compute (RTX 5060 Ti: ~20-30 it/s)

---

## Common Issues

### "Computing KNN K=16" Appears Twice

**This is NORMAL** - Two separate KNN computations:
1. **Training KNN** - At model initialization (cached)
2. **Mesh Extraction KNN** - After training (not cached)

See "Why Two Separate KNN Stages?" section above.

### Iteration Counter Reset Bug (Fixed)

**Symptom:** Training starts from iteration 7K even when loading 40K checkpoint

**Cause:** Hardcoded `iteration = 7000 - 1` in coarse_density.py:533

**Fix Applied:** `iteration = iteration_to_load` (backup_20260128_182509)

**Impact:** Now properly resumes from checkpoint iteration

### Progress Bar Updates Slowly

**Current:** Updates once per batch (~141 images, ~5-10 seconds)

**Expected:** Updates every 10 iterations (~0.3-0.5 seconds)

**Status:** Planned fix in [REFACTOR_SUGAR_PROGRESS_BAR.md](../NOGIT/REFACTOR_SUGAR_PROGRESS_BAR.md)

### Colored Output Not Working (Fixed)

**Symptom:** KNN messages show `[cyan]Computing KNN...[/cyan]` literally

**Cause:** Using `print()` instead of Rich Console

**Fix Applied:** Import Rich Console and use `CONSOLE.print()` (backup_20260128_182741)

**Impact:** Colored output now works for KNN and radius initialization messages

### TensorBoard Data Not Persisting (Fixed)

**Symptom:** Only 1 data point logged at iteration 46,600 instead of continuous data throughout training (40K→65K)

**Root Cause:** 
- TensorBoard only logged every 200 iterations (only 125 data points for 25K iterations)
- Writer buffered data in memory without frequent flushing to disk
- If process interrupted or crashed, buffered data lost

**Fix Applied (backup_20260128_191405):**
- Changed logging frequency: 200 → 10 iterations (2,500 data points for 25K iterations)
- Flush every 10 iterations (after each log batch) instead of every 100
- Flush after every checkpoint save for guaranteed persistence

**Impact:** 
- 20× more data points for detailed loss curves
- No data loss even if training interrupted
- Better visualization of training dynamics

---

## Files Modified (Recent Fixes)

### coarse_density.py
- **Fix:** Iteration resume bug (iteration = iteration_to_load)
- **Backup:** `backup_20260128_182509`
- **Line:** 533

### sugar_model.py
- **Fix:** Colored output (Rich Console integration)
- **Backup:** `backup_20260128_182741`
- **Lines:** 1 (import), 250, 265, 340, 345

---

## References

- **Usage Guide:** [SUGAR_USAGE.MD](SUGAR_USAGE.MD) - Command-line arguments and examples
- **Optimizations:** [SUGAR_OPTIMISATIONS.MD](SUGAR_OPTIMISATIONS.MD) - VRAM and speed improvements
- **Progress Bar Fix:** [REFACTOR_SUGAR_PROGRESS_BAR.md](../NOGIT/REFACTOR_SUGAR_PROGRESS_BAR.md) - Planned improvement

---

**Last Updated:** January 28, 2026  
**Version:** Post-iteration-fix, post-colored-output-fix
