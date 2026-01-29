# SuGaR Usage Guide - Complete Parameter Reference

## Overview

SuGaR (Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction) extracts high-quality textured meshes from trained Gaussian Splatting models.

**Key Insight:** SuGaR requires BOTH:
1. Original COLMAP dataset (`-s`)
2. Trained Gaussian Splatting model (`--gs_output_dir`)

---

## Quick Start

### Best Quality (16GB VRAM Safe)
```bash
conda activate rtx5000_fresh
cd SuGaR/

# Using 60K checkpoint (recommended for best quality)
python train.py \
  -s ../SAMPLES/garden \
  -c ../SAMPLES/garden_output/garden-r2-60k-6M-quality \
  -i 60000 \
  -r dn_consistency \
  --high_poly True \
  --refinement_time long \
  --experiment_name "production-v1" \
  --coarse_iterations 65000 \
  --checkpoint_interval 1000 \
  --checkpoint_milestones 61000 63000 65000 \
  --test_iterations 61000 63000 65000 \
  --export_ply True \
  --eval True \
  --delete_first
```

**Output:** `../SAMPLES/garden_output/garden-r2-60k-6M-quality_mesh/production-v1/`

**Why this is TOP quality:**
- ✅ 60K checkpoint = optimal Gaussian quality (6M Gaussians, 26.77 dB PSNR)
- ✅ `dn_consistency` = best regularization method
- ✅ `--high_poly True` = 1M vertices, 1 Gaussian/triangle
- ✅ `long` = 15000 refinement iterations (best quality)
- ✅ `--experiment_name` = organized output with easy TensorBoard comparison
- ✅ ~1.5-2 hours total time

**Checkpoint Iteration Formula:**
```
--coarse_iterations = checkpoint_iteration + desired_sugar_training_steps

Examples:
-i 7000  --coarse_iterations 15000   # 8K SuGaR iterations (default)
-i 30000 --coarse_iterations 35000   # 5K SuGaR iterations
-i 60000 --coarse_iterations 65000   # 5K SuGaR iterations
```

### Quick Test (Fast)
```bash
python train_full_pipeline.py \
  -s <path_to_colmap_dataset> \
  --gs_output_dir <path_to_trained_gaussians> \
  -r dn_consistency \
  --high_poly True \
  --refinement_time short
```

---

## Full Pipeline Parameters

### Required Arguments

#### `-s, --scene_path`
Path to COLMAP dataset (images + sparse reconstruction)
```bash
-s /path/to/dataset
```

#### `-c, --checkpoint_path`
Path to trained Gaussian Splatting model directory
```bash
-c /path/to/mip_output/<experiment_name>
```
- Must contain `point_cloud/iteration_XXXX/point_cloud.ply`
- Use `-i` to specify which iteration to load (default: 7000)

**Note:** Use `-c` (checkpoint_path) with `train.py`, but `--gs_output_dir` with `train_full_pipeline.py`

#### `-i, --iteration_to_load`
Which checkpoint iteration to load from trained model
```bash
-i 7000   # Load 7K checkpoint
-i 30000  # Load 30K checkpoint
-i 60000  # Load 60K checkpoint (recommended for best quality)
```

**Default:** 7000

**Important:** SuGaR training starts FROM this iteration, so you must set `--coarse_iterations` appropriately:
- `-i 7000` → `--coarse_iterations 15000` (8K SuGaR iterations, default)
- `-i 30000` → `--coarse_iterations 35000` (5K SuGaR iterations)
- `-i 60000` → `--coarse_iterations 65000` (5K SuGaR iterations)

---

### Workflow & Organization

#### `--experiment_name`
Name for this training run (creates organized output structure)

```bash
--experiment_name "production-v1"
--experiment_name "test-phase1"
--experiment_name "6M-quality-5Mverts"
```

**Output Structure:**
```
checkpoint_path_mesh/
├── production-v1/           # Your experiment
│   ├── sugarcoarse_.../
│   │   └── tensorboard/
│   └── mesh files
├── test-phase1/             # Another experiment
│   └── ...
└── 6M-quality-5Mverts/      # Another experiment
    └── ...
```

**Benefits:**
- ✅ Organize multiple experiments per checkpoint
- ✅ Easy TensorBoard comparison (all experiments in parent directory)
- ✅ Consistent with mip-splatting workflow

**TensorBoard:** Automatically shows command for parent directory to compare all experiments:
```bash
tensorboard --logdir "../SAMPLES/garden_output/checkpoint_path_mesh" --port 6007
```

#### `--delete_first`
Automatically delete existing output folder without prompting

```bash
--delete_first  # No Y/n prompt, auto-deletes if folder exists
```

**Use Case:**
- Essential for `conda run` (no stdin for prompt)
- Automated workflows and scripts
- Re-running experiments

**Example with conda run:**
```bash
conda run -n rtx5000_fresh python train.py \
  -s dataset \
  -c checkpoint \
  --experiment_name "test" \
  --delete_first  # Required for conda run!
```

#### `--skip_training`
Skip coarse training and jump straight to mesh extraction (requires `--resume_checkpoint`)

```bash
--skip_training True \
--resume_checkpoint path/to/checkpoint.pt
```

**Use Case:**
- Testing mesh extraction/refinement without waiting for training
- Resuming from checkpoint to test Step 3+ (mesh extraction, refinement, texture)
- Debugging mesh extraction issues
- Iterating on mesh parameters quickly

**Behavior:**
- ✅ Skips slow coarse training phase (Step 2)
- ✅ Uses existing checkpoint from `--resume_checkpoint`
- ✅ Keeps existing output folder (no delete prompt, no deletion)
- ✅ Goes straight to mesh extraction (Step 3)
- ✅ Continues to refinement and texture extraction

**Example:**
```bash
python train.py \
  -s ../SAMPLES/garden \
  -c ../SAMPLES/garden_output/garden-r2-60k-6M-quality \
  -i 40000 \
  -r dn_consistency \
  --resume_checkpoint ../SAMPLES/garden_output/garden-r2-60k-6M-quality_mesh/experiment/sugarcoarse_.../40000.pt \
  --skip_training True \
  --high_poly True \
  --export_ply True
```

**Time Savings:**
- Without: Hours of training (especially at high iterations with regularization)
- With: Starts mesh extraction immediately (~1-2 minutes to begin)

#### `--coarse_iterations`
Total iterations for coarse training phase

```bash
--coarse_iterations 15000  # Default (for 7K checkpoint)
--coarse_iterations 35000  # For 30K checkpoint (5K training)
--coarse_iterations 65000  # For 60K checkpoint (5K training)
```

**Formula:** `checkpoint_iteration + desired_sugar_training_steps`

**Default:** 15000

---

### Quality Control Parameters

#### `-r, --regularization_type`
Surface regularization method (CRITICAL for quality)

**Options:**
- `density` - **RECOMMENDED for 16GB VRAM** - Density-based regularization (73% VRAM, production quality)
- `dn_consistency` - **RECOMMENDED for 24GB+ VRAM** - Depth-normal consistency (best quality, higher VRAM)
- `geometry` - Basic geometric smoothing (fastest, lowest quality)
- `sdf` - Signed Distance Field regularization (experimental)

**Comparison:**

| Mode | Quality | VRAM Usage | Speed | Use Case |
|------|---------|------------|-------|----------|
| `density` | **Excellent** | **~12 GB (73%)** | **1.0-1.2 it/s** | **16GB VRAM (5060 Ti, 5070, 5080)** ✅ |
| `dn_consistency` | **Best** | ~15-18 GB | 0.8-1.0 it/s | **24GB+ VRAM (5090, A5000+)** |
| `geometry` | Fair | Lowest | Fastest | Quick tests only |
| `sdf` | Experimental | Variable | Slow | Research/testing |

**Example:**
```bash
# For 16GB VRAM (RTX 5060 Ti, 5070, 5080)
-r density  # 73% VRAM, excellent quality

# For 24GB+ VRAM (RTX 5090, A5000+)
-r dn_consistency  # Best quality with depth-normal consistency
```

**Performance (Garden dataset, RTX 5060 Ti 16GB):**
- **density:** 73% VRAM (~11.7 GB), 1.0-1.2 it/s, ~2-3 hours for 10K iterations
- **dn_consistency:** Requires 24GB+ VRAM for stable training

#### `--refinement_time`
Training duration for mesh refinement

**Options:**
- `short` - 2,000 iterations (~10-15 min) - Quick test
- `medium` - 7,000 iterations (~30-45 min) - **RECOMMENDED**
- `long` - 15,000 iterations (~1-1.5 hours) - Best quality

**Default:** None (must specify or use `--refinement_iterations` directly)

**Example:**
```bash
--refinement_time long  # Best quality for final output
```

---

### Mesh Density Parameters

#### `--high_poly`
Control mesh decimation

**Options:**
- `True` - Keep full resolution mesh (1M+ vertices)
- `False` - Decimate to lower poly count

**Example:**
```bash
--high_poly True  # Keep all detail
```

#### `--n_vertices_in_fg`
Target vertex count for foreground mesh

**Default:** 1,000,000

**Recommendations:**
- `200000` - Low poly (fast, lower quality)
- `500000` - Medium poly (balanced)
- `1000000` - High poly (default, recommended)
- `2000000` - Ultra high poly (slow, best quality, may need 24GB+ VRAM)

**Example:**
```bash
-v 1000000  # 1M vertices (or use --high_poly True preset)
```

**Note:** Use `-v` or `--n_vertices_in_mesh`, not `--n_vertices_in_fg` (old parameter name)

#### `--square_size`
Resolution of sugar density field

**Default:** 0.01 (cm-scale precision)  
**Lower values** = higher resolution, more memory  
**Higher values** = lower resolution, less detail

```bash
--square_size 0.01  # Default, works well
```

#### `-g, --gaussians_per_triangle`
Number of Gaussian primitives assigned to each mesh triangle

**Default:** 1 (RECOMMENDED)

**Options:**
- `1` - One Gaussian per triangle (cleanest, most efficient) ✓
- `3` - Three Gaussians per triangle (more detail)
- `6` - Six Gaussians per triangle (legacy, diminishing returns)

**What it does:**
- After mesh extraction, SuGaR binds Gaussian primitives to triangle faces
- Each triangle gets assigned N Gaussians to represent its surface appearance
- These Gaussians are then refined to align with the mesh surface

**Quality vs Performance:**

| Value | Gaussians (1M mesh) | Quality | Speed | VRAM |
|-------|---------------------|---------|-------|------|
| **1** | 1M | Excellent | Fastest | Lowest ✓ |
| **3** | 3M | Marginal improvement | Medium | Medium |
| **6** | 6M | Diminishing returns | Slow | High |

**Example:**
```bash
-g 1  # Recommended for most scenes
```

**Why 1 is best:**
- ✅ Cleanest mesh-Gaussian alignment
- ✅ Faster refinement and rendering
- ✅ Easier to work with in game engines (Unity, Unreal)
- ✅ Modern regularization (`dn_consistency`) makes multi-Gaussian less necessary

**When to use higher values:**
- Very fine surface details (fabric weave, rough stone textures)
- Scenes where single Gaussian per triangle looks too smooth
- Legacy compatibility with older SuGaR papers

**Recommendation:** Keep at default `1` unless you have specific texture detail requirements.

---

### Mesh Extraction Parameters

#### `--project_mesh_on_surface_points`
**Default:** `True` (RECOMMENDED)

Refines mesh quality by projecting vertices onto actual surface points extracted from the scene.

**How it works:**
- After generating the initial mesh, finds the nearest surface point for each mesh vertex using k-nearest neighbors
- "Snaps" mesh vertices to these high-quality surface samples
- Transfers both position and color from surface points to mesh
- Results in better detail preservation, especially for fine geometric features

**Options:**
- `True` - Project mesh onto surface points (best quality) ✓
- `False` - Use raw mesh vertices (faster, lower detail)

**Example:**
```bash
--project_mesh_on_surface_points True  # Recommended
```

**Recommendation:** Keep as `True` for production. Only disable for quick tests.

---

#### `--use_custom_bbox`
**Default:** `False` (automatic bounds)

Determines whether to use user-specified bounding box coordinates or automatic camera-based bounds.

**False (default) - Automatic bounds:**
- Computes bounding box based on camera positions and spatial extent
- Foreground: ±`fg_bbox_factor` × camera spatial extent  
- Background: ±`bg_bbox_factor` × camera spatial extent
- Works well for most standard scenes

**True - Manual bounds:**
- Uses user-specified `--bboxmin` and `--bboxmax` coordinates
- Format: `"(x, y, z)"` as strings

**Example:**
```bash
# Manual bounding box
--bboxmin "(0.0, 0.0, 0.0)" \
--bboxmax "(1.0, 1.0, 1.0)"
```

**Use cases:**
- `False`: Standard scenes where automatic bounds work well ✓
- `True`: Extreme camera positions, specific region extraction, or precise volume control

---

#### `--use_centers_to_extract_mesh`
**Default:** `False` (KEEP AS DEFAULT)

Controls whether to use Gaussian centers directly or use proper surface-level sampling.

**False (recommended) - Surface-level extraction:**
- Renders scene from multiple viewpoints
- Extracts surface points at specific density levels (default 0.3)
- Creates accurate surface representation from multiple observations
- **THIS IS THE CORRECT METHOD** ✓

**True (ablation only) - Gaussian centers:**
- Simply uses 3D positions of Gaussian primitives directly
- Much faster but produces **poor quality meshes**
- Useful only for quick tests or ablation studies

**Example:**
```bash
--use_centers_to_extract_mesh False  # Always use this
```

**⚠️ WARNING:** Code explicitly warns: *"Results will look bad, this is not the best way to extract a mesh."*

**Recommendation:** **ALWAYS keep as `False`** for production use.

---

#### `--use_marching_cubes`
**Default:** `False` (Poisson reconstruction)

Chooses between marching cubes algorithm or Poisson surface reconstruction.

**False (recommended) - Poisson Surface Reconstruction:**
- Creates mesh from oriented point cloud (points + normals)
- Produces smooth, watertight surfaces
- Better for organic shapes and general scenes
- Default `poisson_depth=10` controls detail level
- **Faster and smoother results** ✓

**True - Marching Cubes:**
- Computes density on a 3D grid
- Extracts isosurface at specified density threshold
- Good for volumetric data but computationally expensive
- Requires uniform grid evaluation (slower)
- More accurate to density field but less smooth

**Example:**
```bash
--use_marching_cubes False  # Recommended for most scenes
```

**Tradeoffs:**
- **Poisson (False):** Faster, smoother, better for most scenes ✓
- **Marching Cubes (True):** Volumetric accuracy but slower

**Recommendation:** Keep as `False` unless you specifically need volumetric mesh extraction.

---

#### `--surface_level`
**Default:** `0.3`

Surface level threshold for mesh extraction (only used with surface-level extraction).

Controls where in the density field to extract the mesh surface.

**Range:** 0.0 to 1.0
- Lower values (0.1-0.2): Thinner, more conservative surface
- Default (0.3): Balanced surface representation ✓
- Higher values (0.4-0.5): Thicker, more generous surface

**Example:**
```bash
-l 0.3  # or --surface_level 0.3
```

---

### Recommended Mesh Extraction Configuration

For **best quality** (current settings):
```bash
--project_mesh_on_surface_points True   # ✓ Best quality details
--use_centers_to_extract_mesh False     # ✓ Proper surface sampling  
--use_marching_cubes False              # ✓ Poisson reconstruction
-l 0.3                                  # ✓ Balanced surface level
```

This configuration:
- ✅ Maximizes mesh quality and detail preservation
- ✅ Uses proper surface sampling methods
- ✅ Creates smooth, watertight meshes
- ✅ Balances quality, speed, and ease of use
- ✅ **Recommended for production work**

---

### Training Control Parameters

#### `--checkpoint_interval`
Save checkpoint every N iterations

**Default:** 1000  
**Options:** Set to 0 to disable interval checkpoints

**Example:**
```bash
--checkpoint_interval 1000  # Save every 1000 iterations
--checkpoint_interval 0     # Disable interval checkpoints
```

#### `--checkpoint_milestones`
Specific iterations at which to save checkpoints (in addition to interval)

**Default:** `[7000, 9000, 12000, 15000]`

**Example:**
```bash
--checkpoint_milestones 7000 10000 15000
```

#### `--test_iterations`
Iterations at which to run test evaluation and log to TensorBoard

**Default:** `[7000, 10000, 15000]`

**Example:**
```bash
--test_iterations 1000 5000 10000 15000
```

#### `--resume_checkpoint`
Resume training from a SuGaR checkpoint file

**Example:**
```bash
--resume_checkpoint path/to/checkpoint/15000.pt
```

---

### Output Control Parameters

#### `--export_ply`
Export mesh as PLY format

```bash
--export_ply True  # Also saves OBJ with textures
```

#### `-o, --output_dir`
Custom output directory

**Default:** `output/` in SuGaR root  
**Example:**
```bash
-o /path/to/custom_output
```

---

### Evaluation Parameters

#### `--eval`
Enable train/test split evaluation

**Default:** `True` (enabled)
**Options:**
- `True` - Use train/test split for proper generalization metrics
- `False` - Use all images for training (no evaluation)

**Example:**
```bash
--eval True  # Recommended for quality validation
```

**Note:** When enabled, SuGaR will:
- ✅ Split dataset into train/test sets (same split as mip-splatting)
- ✅ Evaluate PSNR, SSIM, LPIPS on test views during training
- ✅ Log metrics to TensorBoard at specified `--test_iterations`

---

#### `--enable_lpips`
Enable LPIPS (Learned Perceptual Image Patch Similarity) metric during evaluation

**Default:** `True` (enabled)  
**Options:**
- `True` - Calculate LPIPS perceptual metric (~1min extra per evaluation)
- `False` - Skip LPIPS (faster evaluation)

**Benefits of LPIPS:**
- ✅ Perceptual similarity metric (standard in 3D reconstruction papers)
- ✅ Detects visual artifacts missed by PSNR/SSIM
- ✅ Correlates better with human quality perception
- ✅ Uses pre-trained VGG network (~500MB download on first use)

**Performance Impact:**
- First evaluation: +2-3 minutes (downloads VGG weights)
- Subsequent: +30-90 seconds per evaluation
- Weights cached at `~/.cache/torch/hub/checkpoints/`

**Example:**
```bash
# Enable (default)
--enable_lpips True

# Disable for faster iteration during experimentation
--enable_lpips False
```

**Recommendation:** Keep enabled for final training runs. Disable only for rapid prototyping.

**Metrics Overview:**
- **L1 Loss**: Pixel-wise difference (lower = better)
- **PSNR**: Peak signal-to-noise ratio in dB (higher = better, 27-30 typical)
- **SSIM**: Structural similarity 0-1 (higher = better, >0.85 typical)
- **LPIPS**: Perceptual distance (lower = better, <0.20 typical)

---

#### `--eval_split`
Which dataset split to evaluate

**Options:**
- `train` - Training views (check overfitting)
- `test` - Test views (validate generalization)

**Example:**
```bash
--eval_split train  # Compare to training PSNR
```

---

### GPU Optimization Parameters

#### `--full_res_normals`
Use full resolution for depth-normal consistency rendering

**Default:** `True` (full-res with gradient checkpointing, max quality)  
**Options:**
- `True` - Full resolution normals with gradient checkpointing (recommended for 16GB VRAM) ✓
- `False` - Half resolution normals (emergency fallback, ~1-3% quality trade-off)

**Example:**
```bash
--full_res_normals True   # Default: Full-res + checkpointing, best quality
--full_res_normals False  # Fallback: Half-res if still OOM
```

**Note:** With Phase 2 gradient checkpointing, full-res normals now work on 16GB VRAM!

#### `--use_gradient_checkpointing`
Enable gradient checkpointing (Phase 2 optimization)

**Default:** `True` (recompute activations, saves 30-40% VRAM)  
**Options:**
- `True` - Gradient checkpointing enabled (recommended for 16GB VRAM) ✓
- `False` - No checkpointing (24GB+ VRAM, 30-50% faster training)

**Example:**
```bash
--use_gradient_checkpointing True   # Default: Max VRAM savings
--use_gradient_checkpointing False  # 24GB+ VRAM: Max speed
```

**Trade-off:** 30-50% slower training for 30-40% VRAM savings

---

#### `--gpu`
GPU device ID (for multi-GPU systems)

```bash
--gpu 0  # Use first GPU
```

#### `--batch_size`
Rendering batch size

**Default:** 1 (safest for 16GB VRAM)  
**Higher values:** Faster but more VRAM

```bash
--batch_size 1  # Safe for 16GB GPUs
```

---

## Quality Parameters Comparison

Complete guide to maximizing mesh quality at different VRAM levels:

| Parameter | Default | Good | Best (16GB) | Ultra (24GB+) |
|-----------|---------|------|-------------|---------------|
| **Gaussians** | 7K checkpoint | 30K checkpoint | **40K checkpoint** | 40K checkpoint |
| **Regularization** | geometry | **dn_consistency** | **dn_consistency** | **dn_consistency** |
| **Vertices** | 1M | 1M | **1M** | **2M** |
| **Gaussians/triangle** | 1 | **1** | **1** | **1** |
| **Refinement time** | medium (7K) | long (15K) | **long (15K)** | **long (15K)** |
| **Coarse iterations** | 15K | 15K | **15K** | **20K** |
| **Full-res normals** | False | False | **False** | **True** |
| **Project mesh** | True | **True** | **True** | **True** |
| **Surface extraction** | Poisson | **Poisson** | **Poisson** | **Poisson** |
| **Time** | ~45 min | ~1.5 hr | **~1.5-2 hr** | **~2-3 hr** |
| **Output PSNR** | ~26-27 dB | ~27-28 dB | **~27-29 dB** | **~28-30 dB** |

**Key Insights:**
- ✅ **Best (16GB) hits maximum quality-per-VRAM ratio** - Ultra only gains ~0.5-1 dB for 2× VRAM
- ✅ **40K checkpoint crucial** - Quality plateaus after 35-40K iterations
- ✅ **dn_consistency required** - Geometry regularization produces rough meshes
- ✅ **15K refinement iterations optimal** - Diminishing returns after
- ⚠️ **Coarse iterations >15K unnecessary** - No measurable quality improvement

---

## Recommended Presets

### Quick Test (10-15 min)
```bash
python train_full_pipeline.py \
  -s /path/to/dataset \
  --gs_output_dir /path/to/trained_model \
  -r dn_consistency \
  --high_poly False \
  --n_vertices_in_fg 200000 \
  --refinement_time short \
  --export_ply True
```

### Balanced Quality (30-45 min)
```bash
python train_full_pipeline.py \
  -s /path/to/dataset \
  --gs_output_dir /path/to/trained_model \
  -r dn_consistency \
  --high_poly True \
  --n_vertices_in_fg 1000000 \
  --refinement_time default \
  --checkpoint_iterations 1000 3500 7000 \
  --test_iterations 1000 3500 7000 \
  --export_ply True \
  --eval_split train
```

### Best Quality - 16GB VRAM (2-3 hours) ⭐ RECOMMENDED
```bash
python train.py \
  -s ../SAMPLES/garden \
  -c ../SAMPLES/garden_output/garden-r2-60k-6M-quality \
  -i 40000 \
  -r density \
  --high_poly True \
  --refinement_time long \
  --experiment_name "production-r2-density" \
  --coarse_iterations 50000 \
  --checkpoint_interval 2000 \
  --checkpoint_milestones 45000 48000 50000 \
  --test_iterations 48000 50000 \
  --export_ply True \
  --eval True \
  --delete_first
```

**What you get:**
- 1M vertex mesh with excellent surface quality
- PSNR: ~27-28 dB (production quality)
- 6M Gaussians from 40K checkpoint
- VRAM: 73% (~11.7 GB on 16GB card)
- Speed: 1.0-1.2 it/s
- Full TensorBoard metrics validation

### Ultra Quality - 24GB+ VRAM (3-4 hours)
```bash
python train.py \
  -s ../SAMPLES/garden \
  -c ../SAMPLES/garden_output/garden-r2-60k-6M-quality \
  -i 40000 \
  -r dn_consistency \
  --high_poly True \
  --refinement_time long \
  --experiment_name "production-r2-dn" \
  --coarse_iterations 50000 \
  --checkpoint_interval 2000 \
  --checkpoint_milestones 45000 48000 50000 \
  --test_iterations 48000 50000 \
  --export_ply True \
  --eval True \
  --delete_first
```

**What you get:**
- 1M vertex mesh with depth-normal consistency (best quality)
- PSNR: ~28-29 dB (maximum quality)
- Requires 24GB+ VRAM (RTX 5090, A5000+)

### Ultra Quality (2-3 hours) - Requires 24GB+ VRAM
```bash
python train.py \
  -s /path/to/dataset \
  -c /path/to/trained_model \
  -i 40000 \
  -r dn_consistency \
  -v 2000000 \
  -g 1 \
  --refinement_time long \
  --full_res_normals True \
  --checkpoint_interval 1000 \
  --checkpoint_milestones 5000 10000 15000 \
  --test_iterations 1000 5000 10000 15000 \
  --export_ply True \
  --eval True
```

---

## Example Commands

### Basic Production Run (Recommended)
```bash
cd SuGaR/

python train.py \
  -s ../SAMPLES/garden \
  -c ../mip-splatting/output/garden-r2-7k/point_cloud/iteration_7000/point_cloud.ply \
  -i 7000 \
  -r dn_consistency \
  --high_poly True \
  --refinement_time medium \
  --test_iterations 1000 3500 7000 10000 15000 \
  --checkpoint_milestones 7000 10000 15000 \
  --export_ply True \
  --eval True
```

**What this does:**
- Uses depth-normal consistency regularization (best quality)
- 1M vertices, 1 Gaussian per triangle (high_poly preset)
- Medium refinement: 7000 iterations (~35-45 min)
- Evaluates at: 1K, 3.5K, 7K, 10K, 15K iterations
- Saves checkpoints at: 7K, 10K, 15K iterations
- Exports textured mesh (.obj) and point cloud (.ply)

### Quick Test (Fast)
```bash
python train.py \
  -s ../SAMPLES/garden \
  -c ../mip-splatting/output/garden-r2-7k/point_cloud/iteration_7000/point_cloud.ply \
  -i 7000 \
  -r dn_consistency \
  --low_poly True \
  --refinement_time short \
  --test_iterations 1000 2000 \
  --export_ply True
```

**What this does:**
- 200K vertices, 6 Gaussians per triangle (low_poly preset)
- Short refinement: 2000 iterations (~10-15 min)
- Quick quality check before full run

### Best Quality (Long)
```bash
python train.py \
  -s ../SAMPLES/garden \
  -c ../mip-splatting/output/garden-r2-7k/point_cloud/iteration_7000/point_cloud.ply \
  -i 7000 \
  -r dn_consistency \
  -v 2000000 \
  -g 1 \
  --refinement_iterations 15000 \
  --full_res_normals True \
  --test_iterations 1000 5000 10000 15000 \
  --checkpoint_milestones 5000 10000 15000 \
  --checkpoint_interval 2000 \
  --export_ply True
```

**What this does:**
- 2M vertices (ultra high poly)
- Long refinement: 15000 iterations (~1-1.5 hours)
- Full resolution normals (requires 24GB+ VRAM)
- Checkpoint every 2000 iterations + milestones
- Best possible quality

### Resume from Checkpoint
```bash
python train.py \
  -s ../SAMPLES/garden \
  -c ../mip-splatting/output/garden-r2-7k/point_cloud/iteration_7000/point_cloud.ply \
  -i 7000 \
  -r dn_consistency \
  --high_poly True \
  --refinement_time long \
  --resume_checkpoint output/refined/15000.pt
```

### With Custom Bounding Box
```bash
python train.py \
  -s ../SAMPLES/garden \
  -c ../mip-splatting/output/garden-r2-7k/point_cloud/iteration_7000/point_cloud.ply \
  -i 7000 \
  -r dn_consistency \
  --high_poly True \
  --refinement_time medium \
  --bboxmin "(0.0, 0.0, 0.0)" \
  --bboxmax "(1.0, 1.0, 1.0)"
```

### Low VRAM Mode (16GB GPU)
```bash
python train.py \
  -s ../SAMPLES/garden \
  -c ../mip-splatting/output/garden-r2-7k/point_cloud/iteration_7000/point_cloud.ply \
  -i 7000 \
  -r dn_consistency \
  --high_poly True \
  --refinement_time medium \
  --full_res_normals False \
  --gpu 0
```

**Optimizations for 16GB VRAM:**
- `--full_res_normals False` - Saves 4-5GB VRAM
- Use `--high_poly True` (1M vertices) instead of custom 2M
- Medium refinement time is safe

---

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir=output/refined --port 6007
```

**Key metrics:**
- `train/loss` - Total training loss
- `train/psnr` - Peak Signal-to-Noise Ratio on training views
- `test/psnr` - PSNR on test views (generalization quality)
- Training command logged in TEXT tab

### Terminal Output
```
[ITER 1000] Saving checkpoint: output/refined/checkpoints/1000.pt
[ITER 7000] Running evaluation... PSNR: 28.45 dB
[ITER 15000] Training complete. Final PSNR: 29.87 dB
Exporting textured mesh...
```

---

## Output Files

After completion, find outputs in `output/`:

```
output/
├── coarse/
│   └── sugarcoarse_3Dgs7000_sdfestim02_sdfnorm02_level03_decim1000000.ply
├── refined/
│   └── sugarfine_3Dgs7000_sdfestim02_sdfnorm02_level03_decim1000000.ply
└── mesh/
    ├── refined_mesh.ply           # Final mesh (PLY format)
    ├── refined_mesh.obj           # Final mesh (OBJ format)
    └── textures/
        ├── material_0.png         # Texture maps
        └── material_0.mtl         # Material definitions
```

---

## Pipeline Stages Explained

### 1. Coarse Density Training
- Converts Gaussians to volumetric density field
- Estimates SDF (Signed Distance Field)
- Extracts coarse mesh using Marching Cubes
- **Time:** ~20-30 minutes

### 2. SDF Refinement
- Optimizes surface alignment
- Improves normal consistency
- Applies regularization (`-r` parameter)
- **Time (short):** ~10-15 minutes (2,000 iterations)
- **Time (default):** ~30-45 minutes (7,000 iterations)
- **Time (long):** ~1-1.5 hours (15,000 iterations)

### 3. Mesh Extraction
- Final Marching Cubes extraction
- Optional decimation (controlled by `--high_poly`)
- Processes 200K-2M vertices depending on `-v` parameter
- **Time:** ~5-10 minutes

### 4. Texture Extraction
- Projects Gaussian colors onto mesh
- Optimizes texture maps using nvdiffrast (GPU rasterization)
- **With sm_120 compilation = 10-60× faster!**
- Exports OBJ + PNG textures
- **Time:** ~5-10 minutes

### Total Pipeline Time

**Quick Test (short refinement):**
- Coarse: ~25 min
- Refinement: ~12 min
- Mesh + Texture: ~15 min
- **Total: ~50 minutes**

**Balanced Quality (default refinement):**
- Coarse: ~25 min
- Refinement: ~35 min
- Mesh + Texture: ~15 min
- **Total: ~1.25 hours**

**Best Quality (long refinement, 2M vertices):**
- Coarse: ~30 min
- Refinement: ~1.25 hours
- Mesh + Texture: ~20 min
- **Total: ~2-2.5 hours**

*Times based on RTX 5060 Ti with garden dataset (161 images, 4.9M Gaussians)*

---

## Troubleshooting

### Out of Memory
- Reduce `--n_vertices_in_fg` to 500000 or 200000
- Set `--batch_size 1`
- Lower `--square_size` (e.g., 0.02)

### Mesh Too Rough
- Use `-r dn_consistency` instead of `geometry`
- Increase `--refinement_time` to `default` or `long`
- Increase `--n_vertices_in_fg`

### Training Too Slow
- Use `--refinement_time short`
- Set `--high_poly False`
- Reduce `--n_vertices_in_fg` to 200000

### Textures Look Bad
- Check nvdiffrast installed correctly with proper compute capability
- Verify trained Gaussians have good PSNR (>27)
- Ensure original images are high resolution

---

## GPU Optimizations

**Optimizations Applied (when using this pipeline):**
- ✅ nvdiffrast compiled with correct compute capability (10-60× faster texture extraction)
- ✅ pytorch3d compiled with correct compute capability (optimized mesh operations)
- ✅ PyMCubes for fast Marching Cubes
- ✅ TF32 enabled for faster matrix ops
- ✅ CUDA 13.x Blackwell optimizations (RTX 5000 series)

**Expected Performance (RTX 5060 Ti / 16GB):**
- Short refinement: 10-15 minutes
- Default refinement: 30-45 minutes
- Long refinement: 1-1.5 hours
- Texture extraction: 2-5 minutes (thanks to nvdiffrast GPU rasterization!)

---

## Next Steps

1. **Validate mesh quality** - Load in MeshLab or Blender
2. **Check textures** - Open `.obj` file with texture maps
3. **Compare PSNR** - Evaluate against training/test views
4. **Load in Unity/Unreal** - Test collision and rendering
5. **Optimize if needed** - Rerun with different parameters

---

## Advanced: Individual Stage Training

Run pipeline stages separately for more control:

### Stage 1: Coarse Density
```bash
python train_coarse_density.py -s <dataset> --gs_output_dir <model>
```

### Stage 2: Coarse SDF
```bash
python train_coarse_sdf.py -s <dataset> --gs_output_dir <model>
```

### Stage 3: Refinement
```bash
python train_refined.py -s <dataset> --gs_output_dir <model>
```

### Stage 4: Mesh Extraction
```bash
python extract_mesh.py -s <dataset> --gs_output_dir <model>
```

### Stage 5: Textured Mesh
```bash
python extract_refined_mesh_with_texture.py -s <dataset> --gs_output_dir <model>
```

---

**For installation:** [INSTALL.md](../INSTALL.md)  
**For optimizations:** [SUGAR_OPTIMISATIONS.MD](SUGAR_OPTIMISATIONS.MD)  
**For Mip-Splatting training:** [MIPS_TRAIN.MD](MIPS_TRAIN.MD)
