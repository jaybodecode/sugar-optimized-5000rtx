# Mip-Splatting Usage Guide - Complete Parameter Reference

## Overview

Mip-splatting is an enhanced 3D Gaussian Splatting method with 3D filtering to reduce aliasing artifacts. This guide covers all command-line arguments for `train.py`.

---

## Quick Start

```bash
conda activate sugar
cd mip-splatting/

# Basic training (half resolution, 7K iterations)
python train.py -s ../SAMPLES/garden --iteration 7000 -r 2

# Full quality (full resolution, 30K iterations)
python train.py -s ../SAMPLES/garden --iteration 30000 -r 1
```

---

## Command Line Arguments

### Required Arguments

#### `-s, --source_path`
**Type:** `str` (required)  
**Description:** Path to COLMAP dataset directory

Must contain:
- `sparse/0/` - COLMAP reconstruction (cameras, points3D)
- `images/` - Source images (or custom path via `--images`)

**Example:**
```bash
-s /path/to/dataset
-s ../SAMPLES/garden
```

---

### Core Training Parameters

#### `--iteration`
**Type:** `int`  
**Default:** `30000`  
**Description:** Total training iterations

**Recommendations:**
- `7000` - Quick preview (~15-20 min at -r 2)
- `15000` - Good quality for testing
- `30000` - Full training (default, best quality)
- `40000+` - High-res production runs

**Example:**
```bash
--iteration 7000
```

#### `-r, --resolution`
**Type:** `int`  
**Default:** `-1` (auto)  
**Description:** Resolution downsample factor

**Options:**
- `-1` - Auto (uses COLMAP image size)
- `1` - Full resolution (~20 min for 7K iters, best quality)
- `2` - Half resolution (~10-15 min for 7K iters, recommended)
- `4` - Quarter resolution (~5-8 min for 7K iters, fast preview)
- `8` - Eighth resolution (~3-5 min for 7K iters, testing only)

**Example:**
```bash
-r 2  # Half resolution (default balance)
```

**Performance (Garden dataset, 7K iterations on RTX 5060 Ti):**
| Resolution | Pixels | Time | Speed | Use Case |
|------------|--------|------|-------|----------|
| `-r 1` | ~5187×3361 | ~20 min | 5-6 it/s | Production |
| `-r 2` | ~2594×1681 | ~10 min | 5-7 it/s | Recommended ✅ |
| `-r 4` | ~1297×840 | ~5 min | 8-10 it/s | Fast preview |
| `-r 8` | ~649×420 | ~3 min | 12-15 it/s | Testing only |

---

### Output Control

#### `-m, --model_path`
**Type:** `str`  
**Default:** Auto-generated next to dataset  
**Description:** Output directory for trained model

**Default behavior:**
- Creates `<dataset>_output/<experiment_name>/` or `<dataset>_output/<timestamp>/`
- Keeps output organized next to source data

**Example:**
```bash
-m /custom/output/path
```

#### `--experiment_name`
**Type:** `str`  
**Default:** `None` (uses timestamp)  
**Description:** Name for this training run

Appended to output path for easy identification.

**Example:**
```bash
--experiment_name "garden-r2-7k"
# Output: ../SAMPLES/garden_output/garden-r2-7k/
```

#### `--delete_first`
**Type:** `bool` (flag)  
**Default:** `False`  
**Description:** Auto-delete existing output folder without prompting

**Behavior:**
- **Without flag:** Prompts user "Delete existing folder? [y/N]" if output path exists
- **With flag:** Automatically deletes and continues (useful for scripts/automation)

**Use cases:**
- **Batch training scripts** - Avoid interactive prompts blocking execution
- **Automated pipelines** - CI/CD or scheduled training runs
- **Development iteration** - Quick re-runs without manual confirmation

**Example:**
```bash
# Re-run same experiment without prompt
python train.py -s ../SAMPLES/garden -r 2 --experiment_name "test-run" --delete_first

# Batch script example
for scene in garden bicycle; do
    python train.py -s ../SAMPLES/$scene -r 2 --delete_first --experiment_name "$scene-batch"
done
```

**Warning:** Use carefully - deletes without confirmation! Consider backing up important results first.

---

#### `--test_iterations`
**Type:** `int[]`  
**Default:** `[7000, 30000]`  
**Description:** Iterations to run evaluation (PSNR, L1 loss)

Renders test cameras and computes metrics.

**Example:**
```bash
--test_iterations 1000 5000 7000
```

#### `--save_iterations`
**Type:** `int[]`  
**Default:** `[7000, 30000]`  
**Description:** Iterations to save final `.ply` Gaussians

**Example:**
```bash
--save_iterations 5000 7000
```

#### `--checkpoint_iterations`
**Type:** `int[]`  
**Default:** `[]` (none)  
**Description:** Iterations to save resumable `.pth` checkpoints

Allows resuming training from these points.

**Example:**
```bash
--checkpoint_iterations 5000 10000 15000
```

#### `--start_checkpoint`
**Type:** `str`  
**Default:** `None`  
**Description:** Path to checkpoint to resume from

**Example:**
```bash
--start_checkpoint ./output/chkpnt15000.pth
```

---

### Scene Parameters

#### `--images`
**Type:** `str`  
**Default:** `"images"`  
**Description:** Subdirectory name for images within dataset

**Example:**
```bash
--images "images_2"  # Use <dataset>/images_2/ instead of <dataset>/images/
```

#### `--sh_degree`
**Type:** `int`  
**Default:** `3`  
**Description:** Spherical harmonics degree (0-3)

Controls view-dependent color complexity:
- `0` - No view dependence (flat colors)
- `1` - Basic view dependence
- `2` - Good quality, less VRAM
- `3` - Best quality (default)

**Example:**
```bash
--sh_degree 2  # Save ~10-15% VRAM
```

#### `--white_background`
**Type:** `flag`  
**Default:** `False` (black)  
**Description:** Use white background instead of black

**Example:**
```bash
--white_background  # For scenes with white/bright backgrounds
```

#### `--eval`
**Type:** `flag`  
**Default:** `False`  
**Description:** Enable train/test split mode

Uses COLMAP's train/test split if available (every 8th image becomes test set).

**Example:**
```bash
--eval  # Required for proper test set evaluation
```

#### `--eval_only`
**Type:** `flag`  
**Default:** `False`  
**Description:** Evaluation-only mode (skips optimizer setup)

**Benefits:**
- Saves 2-3GB VRAM by not creating Adam optimizer states
- Perfect for running metrics on existing checkpoints
- Faster startup (no optimizer initialization)

**Requirements:**
- Must use `--start_checkpoint <path>` to load trained model
- **Important:** Set `--iteration` slightly higher than checkpoint iteration (e.g., checkpoint at 7000 → use `--iteration 7001` or higher)
- Set `--test_iterations` to match `--iteration` to trigger evaluation

**Why iteration must be higher:**
Training loop exits immediately if current iteration equals target iteration. By setting target 1+ iterations higher, the model runs briefly (without optimizer updates) then triggers evaluation.

**Use Cases:**
- Quick quality assessment of existing models
- VRAM-limited systems (16GB or less)
- Batch evaluation of multiple checkpoints
- Testing different evaluation resolutions

**Example:**
```bash
# Evaluate existing checkpoint from iteration 7000
# Note: --iteration 7001 (not 7000) to trigger evaluation
python train.py \
  -s ../SAMPLES/garden \
  --iteration 7001 \
  --start_checkpoint ../SAMPLES/garden_output/model/chkpnt7000.pth \
  --test_iterations 7001 \
  -r 2 \
  --eval_only \
  --experiment_name "eval-metrics"
```

**VRAM Savings:**
| Mode | VRAM Usage (7K iter model) | Notes |
|------|---------------------------|-------|
| Normal | ~16GB | Full optimizer states |
| `--eval_only` | ~13-14GB | No optimizer (2-3GB saved) |
| `--eval_only --enable_lpips False` | ~12-13GB | +500MB saved |
| `--eval_only -r 4` | ~10-11GB | +3-4GB saved from resolution |

---

#### `--test_camera_count`
**Type:** `int`  
**Default:** `6`  
**Description:** Number of cameras for test set (evenly distributed, includes first & last frame)

Directly controls test set size for COLMAP datasets:
- `6` = 6 test cameras (default, VRAM-friendly for 16GB GPUs)
- `8` = 8 test cameras (more evaluation coverage)
- `12` = 12 test cameras (~6.5% for 185-camera datasets)
- `20` = 20 test cameras (~10% split, better statistical validity)

**Distribution strategy:**
- Always includes **first frame** (index 0)
- Always includes **last frame** (index N-1)
- Middle cameras **evenly distributed** between first/last
- Example (6 test cameras from 185 total): indices 0, 37, 74, 111, 148, 184

**Benefits:**
- **Direct control** - exactly N test cameras, no math needed
- **No wasted cameras** - all cameras used for either training or testing
- **Temporal coverage** - evenly distributed across capture sequence
- **VRAM efficient** - fewer test cameras = less memory during evaluation

**VRAM impact during evaluation:**

During test iterations, each test camera simultaneously loads into VRAM:
- **Ground truth image** (uncompressed float32)
- **Rendered output** (uncompressed float32)
- **Depth map** (for depth metric computation)
- **Normal maps** (for geometric validation)
- **LPIPS features** (VGG network activations)

**Memory per image (approximate):**
| Resolution | Per Image | 6 cameras | 20 cameras |
|------------|-----------|-----------|------------|
| 1920×1080 (Full HD) | ~25 MB | ~150 MB | ~500 MB |
| 2594×1680 (Garden) | ~42 MB | ~252 MB | ~840 MB |
| 3840×2160 (4K) | ~100 MB | ~600 MB | ~2 GB |

**VRAM spike = (test_camera_count × per_image_memory) + model + optimizer**

For 16GB GPUs at high resolution:
- Keep `test_camera_count ≤ 6-8` to prevent OOM during evaluation
- For 24GB+ GPUs: `test_camera_count = 12-20` works well
- For 48GB GPUs: `test_camera_count = 30+` for comprehensive validation

**Example:**
```bash
# Default: 6 test cameras (16GB GPU, high-res safe)
python train.py -s ../SAMPLES/garden -r 2 --eval

# More evaluation coverage (24GB GPU)
python train.py -s ../SAMPLES/garden -r 2 --eval --test_camera_count 12

# Benchmark-quality evaluation (48GB GPU or -r 4)
python train.py -s ../SAMPLES/garden -r 2 --eval --test_camera_count 20
```

**Note:** Only affects COLMAP datasets when `--eval` is enabled. Blender datasets use predefined splits from transforms_*.json files.

---

### Mip-Splatting Specific Parameters

#### `--kernel_size`
**Type:** `float`  
**Default:** `0.1`  
**Description:** 3D filter radius for anti-aliasing

Larger = more blur (less aliasing), smaller = sharper (more aliasing).

**Example:**
```bash
--kernel_size 0.15  # More aggressive anti-aliasing
```

#### `--ray_jitter`
**Type:** `flag`  
**Default:** `False`  
**Description:** Enable subpixel ray jittering

Adds random offset [-0.5, 0.5] to each ray for anti-aliasing.  
⚠️ Reduces training speed ~10%. Requires `--resample_gt_image`.

**Example:**
```bash
--ray_jitter --resample_gt_image
```

#### `--resample_gt_image`
**Type:** `flag`  
**Default:** `False`  
**Description:** Resample ground truth with subpixel offset

Required when using `--ray_jitter` to match jittered rays.

---

### Memory & Performance

#### `--low_dram`
**Type:** `flag`  
**Default:** `False` (eager loading)  
**Description:** Enable lazy image loading with LRU cache

**Modes:**
- **Eager (default):** All images loaded at startup (~6-8GB RAM for 185 images, 30-60s startup)
  - ⚠️ **Slow startup:** Single-threaded PIL decode (CPU-bound, not disk I/O)
  - ✅ **Fast training:** Images already in RAM
- **Lazy (`--low_dram`):** Load on-demand with LRU cache (~1-2GB RAM, instant startup)
  - ✅ **Instant startup:** Images loaded as needed during training
  - ⚠️ **Slightly slower first epoch:** Cache warms up during training

**Startup Time (185 images @ -r 2):**
- Eager: 30-60 seconds (PIL decode bottleneck - low disk utilization is normal)
- Lazy: <1 second (deferred loading)

**Example:**
```bash
--low_dram  # Instant startup, use if RAM limited (<16GB) or want fast startup
```

#### `--image_cache_gb`
**Type:** `float`  
**Default:** `2.0`  
**Description:** LRU cache memory limit in GB for lazy loading

Only used with `--low_dram`. Automatically calculates image count based on resolution.

**Recommendations:**
- Low RAM (<16GB): `1.0` GB
- Medium RAM (16-32GB): `2.0` GB (default)
- High RAM (32GB+): `5.0` GB

**Example:**
```bash
--low_dram --image_cache_gb 5.0  # Cache ~102 images at -r 2
```

#### `--data_device`
**Type:** `str`  
**Default:** `"cpu"`  
**Description:** Device for image data storage

**Options:**
- `cpu` - Store images in RAM (default, recommended)
- `cuda` - Store images in VRAM (uses 6-8GB extra VRAM!)

**Example:**
```bash
--data_device cpu  # Default, keeps VRAM free for training
```

#### `--accumulation_steps`
**Type:** `int`  
**Default:** `1`  
**Description:** Gradient accumulation steps

Simulates larger batch size by accumulating gradients over multiple iterations before optimizer step.

**Use cases:**
- Higher stability for difficult scenes
- Simulate batch training on limited VRAM

**Example:**
```bash
--accumulation_steps 4  # 4x gradient accumulation
```

---

### Advanced Training Parameters

#### Densification Parameters

**`--densify_from_iter`**  
**Type:** `int` | **Default:** `500`  
Start densification (split/clone gaussians) at this iteration.

**`--densify_until_iter`**  
**Type:** `int` | **Default:** `8000` (RTX 5060 Ti emergency brake)  
Stop densification at this iteration (refinement only after).

**NEW: Changed from 15000→12000→8000 (emergency brake) to prevent VRAM overflow on 16GB GPUs.**
- RTX 5060 Ti 16GB: Use `8000` (NEW default)
- RTX 4090 24GB+: Use `--densify_until_iter 15000` for full quality

**`--densification_interval`**  
**Type:** `int` | **Default:** `100`  
Iterations between densification operations.

**`--densify_grad_threshold`**  
**Type:** `float` | **Default:** `0.0002`  
Gradient threshold for splitting/cloning gaussians.

Lower = more Gaussians (higher quality, more VRAM):
- `0.0001` - Very dense (~7-10M Gaussians)
- `0.0002` - Default balance (~5M Gaussians)
- `0.0004` - Fewer Gaussians (~3M, less VRAM)

**Example:**
```bash
--densify_grad_threshold 0.0004  # Reduce Gaussian count for 16GB VRAM
```
**`--min_opacity_threshold`**  
**Type:** `float` | **Default:** `0.03` (RTX 5060 Ti optimized)  
**Description:** Opacity threshold for pruning invisible Gaussians during densification.

Gaussians with opacity below this threshold are removed as "junk":
- `0.005` - Conservative pruning (old default, may accumulate invisible Gaussians, risk VRAM overflow on 16GB)
- `0.01` - Moderate pruning (sigmoid ≈ 50.25% opacity cutoff)
- `0.03` - Aggressive pruning (**NEW default**, sigmoid ≈ 51.25%, optimized for RTX 5060 Ti 16GB)
- `0.05` - Very aggressive (use with caution, may affect quality)

**Quality Impact:** Near-zero! These Gaussians contribute <0.25% to pixel colors (effectively invisible).

**VRAM Impact:** Can reduce Gaussian count by 20-30% during training, preventing overflow.

**⚠️ Warning:** Changing this parameter mid-training (via checkpoint resume) causes optimizer conflicts. Start fresh training if modifying.

**Example:**
```bash
--min_opacity_threshold 0.005  # Conservative (for 24GB+ VRAM)
```

---

#### Binary Search Hard Cap (VRAM Protection)

**`--target_gaussian_count`**  
**Type:** `int` | **Default:** `5000000` (5M - RTX 5060 Ti optimized)  
**Description:** Absolute maximum Gaussian count enforced via binary search pruning.

When enabled (`> 0`), automatically finds the exact opacity threshold to keep **exactly** this many Gaussians. Prevents VRAM overflow by enforcing a hard ceiling, regardless of scene complexity.

**Recommended values by GPU VRAM:**
- `5000000` (5M) - **NEW default** for RTX 5060 Ti 16GB, RTX 4060 Ti 16GB
- `8000000` (8M) - RTX 4090 24GB, RTX A5000 24GB
- `12000000` (12M) - RTX 6000 Ada 48GB, A100 40GB
- `0` - Disabled (for 24GB+ GPUs, remove hard cap)

**How it works:**
1. After densify_and_prune, counts current Gaussians
2. If over target, binary search finds threshold to hit **exact** count
3. Removes low-opacity Gaussians until target reached

**Example:**
```bash
--target_gaussian_count 5000000  # Enforce 5M cap for 16GB VRAM (RTX 5060 Ti safe limit)
```

**`--binary_search_min`**  
**Type:** `float` | **Default:** `0.003` (optimized for faster convergence)  
Lower bound for binary search opacity threshold. Start of search range.

**`--binary_search_max`**  
**Type:** `float` | **Default:** `0.08` (more conservative)  
Upper bound for binary search opacity threshold. End of search range.

**`--binary_search_iterations`**  
**Type:** `int` | **Default:** `12` (balanced precision)  
Number of binary search iterations. Controls precision:
- `10` iterations = ~0.001 precision (faster, less precise)
- `12` iterations = ~0.0002 precision (**NEW default**, balanced)
- `15` iterations = ~0.0001 precision (maximum precision)
- `20` iterations = ~0.00001 precision (overkill for most cases)

**Advanced Example:**
```bash
# Fine-tuned binary search for specific scene
--target_gaussian_count 6000000 \
--binary_search_min 0.003 \      # Start search higher (faster convergence)
--binary_search_max 0.08 \        # Lower max (avoids over-pruning)
--binary_search_iterations 12     # Slightly faster, still precise enough
```

**⚠️ Important:** Binary search hard cap works **in addition to** `--min_opacity_threshold`. The threshold parameter prunes during densification, while hard cap enforces absolute ceiling afterward.

---

**`--opacity_reset_interval`**  
**Type:** `int` | **Default:** `3000`  
Reset opacity every N iterations to prevent floaters.

**`--percent_dense`**  
**Type:** `float` | **Default:** `0.01`  
Percentage of scene extent for densification culling.

---

#### Learning Rates

**`--position_lr_init`** = `0.00016` - Initial position learning rate  
**`--position_lr_final`** = `0.0000016` - Final position learning rate  
**`--position_lr_max_steps`** = `30000` - Steps for position LR schedule  
**`--feature_lr`** = `0.0025` - Spherical harmonics learning rate  
**`--opacity_lr`** = `0.05` - Opacity learning rate  
**`--scaling_lr`** = `0.005` - Gaussian scale learning rate  
**`--rotation_lr`** = `0.001` - Rotation (quaternion) learning rate

**Example:**
```bash
--position_lr_init 0.0002  # Faster initial learning
```

---

#### Loss Function

**`--lambda_dssim`**  
**Type:** `float` | **Default:** `0.2`  
Weight for DSSIM (structural similarity) loss.

Total loss = `(1 - λ) * L1 + λ * DSSIM`

- `0.2` - Default (80% L1, 20% DSSIM)
- `0.0` - Pure L1 loss
- `0.5` - Equal L1 and DSSIM

**Example:**
```bash
--lambda_dssim 0.3  # More emphasis on structural similarity
```

---

### High-Resolution Sampling

#### `--sample_more_highres`
**Type:** `flag`  
**Default:** `False`  
**Description:** Sample high-res cameras more frequently

30% chance to sample from cameras with width ≥ 800px.  
Useful for mixed-resolution datasets.

**Example:**
```bash
--sample_more_highres  # For datasets with varying resolutions
```

#### `--load_allres`
**Type:** `flag`  
**Default:** `False`  
**Description:** Load all resolution levels

Experimental feature for multi-resolution training.

---

### Debug & GUI

#### `--quiet`
**Type:** `flag`  
**Default:** `False`  
**Description:** Suppress training output

**Example:**
```bash
--quiet  # Minimal output
```

#### `--debug_from`
**Type:** `int`  
**Default:** `-1` (disabled)  
**Description:** Enable debug mode starting at this iteration

**Example:**
```bash
--debug_from 5000  # Debug starting at iteration 5000
```

#### `--detect_anomaly`
**Type:** `flag`  
**Default:** `False`  
**Description:** Enable PyTorch anomaly detection

Slower but helps catch NaN/Inf errors.

**Example:**
```bash
--detect_anomaly  # For debugging training instability
```

#### `--ip` / `--port`
**Type:** `str` / `int`  
**Default:** `127.0.0.1` / `6009`  
**Description:** Network GUI server settings

For real-time viewer (rarely used).

---

## Complete Example Commands

### Quick Preview (Fast)
```bash
python train.py \
  -s ../SAMPLES/garden \
  --iteration 7000 \
  -r 4 \
  --experiment_name "garden-preview"
```
**Time:** ~5 minutes  
**Use:** Quick quality check

---

### Recommended Training (Balanced)
```bash
python train.py \
  -s ../SAMPLES/garden \
  --iteration 7000 \
  -r 2 \
  --test_iterations 1000 5000 7000 \
  --checkpoint_iterations 5000 \
  --save_iterations 5000 7000 \
  --experiment_name "garden-r2-7k"
```
**Time:** ~10-15 minutes  
**Use:** Best balance for most scenes

---

### Production Quality (Full Resolution)
```bash
python train.py \
  -s ../SAMPLES/garden \
  --iteration 30000 \
  -r 1 \
  --test_iterations 7000 15000 30000 \
  --checkpoint_iterations 7000 15000 20000 \
  --save_iterations 7000 15000 30000 \
  --experiment_name "garden-full-res"
```
**Time:** ~90 minutes  
**Use:** Final high-quality output

---

### Low VRAM Training (16GB GPU)
```bash
python train.py \
  -s ../SAMPLES/garden \
  --iteration 7000 \
  -r 2 \
  --sh_degree 2 \
  --densify_grad_threshold 0.0004 \
  --low_dram \
  --image_cache_size 10 \
  --experiment_name "garden-lowvram"
```
**VRAM:** ~10-12GB peak  
**Use:** RTX 5060 Ti 16GB or similar

---

### Resume from Checkpoint
```bash
python train.py \
  -s ../SAMPLES/garden \
  --iteration 30000 \
  -r 2 \
  --start_checkpoint ../SAMPLES/garden_output/garden-r2/chkpnt15000.pth \
  --experiment_name "garden-r2-resumed"
```

---

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir="../SAMPLES/garden_output" --port 6006 --bind_all
# Open: http://localhost:6006
```

**Key Metrics:**
- `train_loss_patches/l1_loss` - Reconstruction error
- `train_loss_patches/total_loss` - Combined L1 + SSIM
- `test/loss_viewpoint - psnr` - Test set quality (26-30 dB typical)
- `3_gaussians/total_count_K` - Gaussian count (5M typical)
- `4_performance/fps` - Training speed

---

### Terminal Status
Real-time status line updated every 50 iterations:
```
2100/7000 │ L:0.104 │ VRAM:37% (5.9GB) │ RAM:61% (15.3GB) │ CPU:11% │ 5.62it/s │ 6m │ ETA:14m
```

**Phase Announcements:**
- Iteration 500: Densification phase starts
- Iteration 12000: Stable training phase (refinement only)

---

## Training Phases

### Phase 1: Initialization (Iteration 0-500)
- Load COLMAP cameras and sparse points
- Initialize Gaussian parameters
- Speed: 2-4 it/s (warmup)

### Phase 2: Densification (Iteration 500-12000)
- Split/clone/prune Gaussians every 100 iterations
- Gaussian count grows: 50K → 5-6M
- Speed: 4-7 it/s (fluctuates during densification)

### Phase 3: Stable Training (Iteration 12000+)
- No more densification (refinement only)
- Gaussian count stable
- Speed: 6-8 it/s (consistent)

---

## Output Files

After training, output directory contains:

```
<dataset>_output/<experiment_name>/
├── point_cloud/
│   ├── iteration_7000/
│   │   └── point_cloud.ply      # 3D Gaussians (5M points, ~1.2GB)
│   └── iteration_30000/
│       └── point_cloud.ply
├── cameras.json                  # Camera parameters
├── cfg_args                      # Training config (for resuming)
├── chkpnt7000.pth               # Resumable checkpoint (if --checkpoint_iterations used)
└── events.out.tfevents.*        # TensorBoard logs
```

**Use PLY file for:**
- Viewing in SuperSplat viewer
- Mesh extraction with SuGaR
- Further refinement

---

## Tips & Best Practices

### Resolution Selection
- `-r 2` is the sweet spot for most use cases
- `-r 1` for final production only (4× slower)
- `-r 4` for quick iteration during development

### Iteration Count
- 7K iterations is enough for good quality
- 15K-30K for production/research
- Diminishing returns after 30K

### Memory Optimization
1. Use `--low_dram` if RAM < 16GB
2. Use `-r 2` instead of `-r 1` if VRAM limited
3. Reduce `--sh_degree 2` saves 10-15% VRAM
4. Increase `--densify_grad_threshold 0.0004` reduces Gaussian count

### Speed Optimization
1. Use native Linux paths (avoid `/mnt/c` on WSL)
2. Default eager loading is fastest (don't use `--low_dram` unless needed)
3. Lower resolution = faster training

---

**See also:**
- [MIPS_TRAIN.MD](MIPS_TRAIN.MD) - Detailed training phases and troubleshooting
- [SUGAR_USAGE.MD](SUGAR_USAGE.MD) - Mesh extraction from trained Gaussians
- [MIPS_OPTIMISATION.MD](MIPS_OPTIMISATION.MD) - Advanced optimization strategies
