# Mip-Splatting Training Documentation

Complete reference for `train.py` arguments and training phases.

---

## Command Line Arguments

### Model Parameters (`-s`, `-m`, etc.)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-s, --source_path` | str | **required** | Path to dataset (COLMAP format) |
| `-m, --model_path` | str | auto | Output directory for trained model |
| `--sh_degree` | int | 3 | Spherical harmonics degree (0-3) |
| `--images` | str | "images" | Subdirectory name for images |
| `-r, --resolution` | int | -1 | Resolution divisor (1=full, 2=half, 4=quarter, -1=auto) |
| `--white_background` | flag | False | Use white background instead of black |
| `--data_device` | str | "cpu" | Device for image data storage (cpu/cuda) - cpu reduces VRAM pressure |
| `--eval` | flag | False | Enable evaluation mode (train/test split) |
| `--kernel_size` | float | 0.1 | Mip-splatting kernel size (3D filter radius) |
| `--ray_jitter` | flag | False | Enable subpixel ray jittering for anti-aliasing |
| `--resample_gt_image` | flag | False | Resample GT with subpixel offset |
| `--load_allres` | flag | False | Load all resolution levels |
| `--sample_more_highres` | flag | False | Sample high-res images more frequently (30% chance) |
| `--low_dram` | flag | False | **Lazy loading** - load images on-demand with LRU cache (saves RAM, ~2GB) |

**Note:** Eager loading (all images in RAM) is now the default behavior for maximum speed. Use `--low_dram` if you have limited RAM.

### Optimization Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--iterations` | int | 30,000 | Total training iterations |
| `--position_lr_init` | float | 0.00016 | Initial position learning rate |
| `--position_lr_final` | float | 0.0000016 | Final position learning rate |
| `--position_lr_delay_mult` | float | 0.01 | Position LR warmup multiplier |
| `--position_lr_max_steps` | int | 30,000 | Steps for position LR schedule |
| `--feature_lr` | float | 0.0025 | Feature (SH) learning rate |
| `--opacity_lr` | float | 0.05 | Opacity learning rate |
| `--scaling_lr` | float | 0.005 | Scaling learning rate |
| `--rotation_lr` | float | 0.001 | Rotation learning rate |
| `--percent_dense` | float | 0.01 | Percentage of scene extent for densification |
| `--lambda_dssim` | float | 0.2 | Weight for SSIM loss (1-λ for L1 loss) |
| `--densification_interval` | int | 100 | Iterations between densification |
| `--opacity_reset_interval` | int | 3,000 | Iterations between opacity resets |
| `--densify_from_iter` | int | 500 | Start densification at this iteration |
| `--densify_until_iter` | int | 15,000 | Stop densification at this iteration |
| `--densify_grad_threshold` | float | 0.0002 | Gradient threshold for densification |

### Pipeline Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--convert_SHs_python` | flag | False | Use Python for SH conversion (debug) |
| `--compute_cov3D_python` | flag | False | Use Python for covariance (debug) |
| `--debug` | flag | False | Enable debug mode |

### Training Control

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--test_iterations` | int[] | [7000, 30000] | Iterations to run evaluation |
| `--save_iterations` | int[] | [7000, 30000] | Iterations to save model |
| `--checkpoint_iterations` | int[] | [] | Iterations to save checkpoint |
| `--start_checkpoint` | str | None | Path to checkpoint to resume from |
| `--debug_from` | int | -1 | Iteration to start debug mode |
| `--detect_anomaly` | flag | False | Enable PyTorch anomaly detection |
| `--quiet` | flag | False | Suppress output |
| `--ip` | str | 127.0.0.1 | GUI server IP address |
| `--port` | int | 6009 | GUI server port |

---

## Training Phases

### Phase 1: Initialization (Iteration 0)
- Load dataset (COLMAP cameras, images, point cloud)
- Initialize Gaussian parameters from sparse point cloud
- Setup optimizers with learning rate schedules
- Compute initial 3D filters for all training cameras
- **Memory mode**: Eager loading (default, fast) or Lazy (with `--low_dram`, saves RAM)

**Expected behavior:**
- VRAM: 2-4 GB (initial scene)
- GPU utilization: 30-50%
- Duration: 10-30 seconds

---

### Phase 2: Densification (Iterations 500-15,000)

#### Active Processes:
1. **Every 100 iterations** (500-15,000):
   - Densify and prune Gaussians based on gradient threshold
   - Split large Gaussians with high gradients
   - Clone small Gaussians with high gradients
   - Remove Gaussians with low opacity (<0.005)
   - Recompute 3D filters for new Gaussians

2. **Every 1000 iterations**:
   - Increase spherical harmonics degree (up to sh_degree)

3. **Every 3000 iterations**:
   - Reset opacity to prevent Gaussians from becoming too transparent

#### Key Metrics:
- **Gaussian count growth**: ~50K → 500K-2M (depends on scene)
- **GPU utilization**: Fluctuates 40-90% during densify/prune
- **Speed**: 2-7 it/s (slower during densification)

**Densification logic:**
```python
if iteration > 500 and iteration % 100 == 0:
    if iteration < 15000:
        densify_and_prune(grad_threshold=0.0002, opacity_threshold=0.005)
        compute_3D_filter()  # Update filters for new Gaussians
```

---

### Phase 3: Stable Training (Iterations 15,000-30,000)

#### Active Processes:
1. **Standard gradient descent** on all Gaussian parameters
2. **3D filter updates every 100 iterations** (no more densification)
3. **Continued SH degree increases** every 1000 iterations
4. **Periodic evaluation** at test_iterations [7000, 30000]

#### Key Metrics:
- **Gaussian count**: Stable (~500K-2M)
- **GPU utilization**: Consistent 85-100%
- **Speed**: 6-8 it/s (fast and stable)
- **Loss convergence**: Should plateau around iteration 20,000

**Expected behavior:**
- Loss should decrease smoothly
- PSNR should increase to 26-30 dB (scene dependent)
- No major VRAM spikes

---

### Phase 4: Evaluation & Saving

#### Test Iterations [7000, 30000]:
- Render all test views
- Compute L1 loss and PSNR
- Log images to TensorBoard
- Clear CUDA cache after evaluation

#### Save Iterations [7000, 30000, final]:
- Save Gaussian parameters: `point_cloud/iteration_XXXX/point_cloud.ply`
- Save cameras: `cameras.json`
- Save config: `cfg_args`

**Output structure:**
```
output/
├── point_cloud/
│   ├── iteration_7000/
│   │   └── point_cloud.ply
│   └── iteration_30000/
│       └── point_cloud.ply
├── cameras.json
├── cfg_args
└── [tensorboard logs]
```

---

## Memory Management

### RTX GPU Optimizations (Built-in)
```python
# Enabled by default in train.py
torch.backends.cuda.matmul.allow_tf32 = True      # TF32 acceleration
torch.backends.cudnn.allow_tf32 = True            # cuDNN TF32
torch.backends.cudnn.benchmark = True             # Auto-tune kernels

# Automatic memory cleanup every 100 iterations
if iteration % 100 == 0:
    torch.cuda.empty_cache()

# Tensor cleanup after backward pass
del render_pkg, image, gt_image, viewspace_point_tensor
del visibility_filter, radii, Ll1, loss
```

### Image Loading Modes

**Default (Eager Loading)**:
- All images loaded at startup into RAM
- Faster training (no I/O overhead)
- Requires ~8GB RAM for 185 images at full resolution
- **Best for**: Most systems with 16GB+ RAM (recommended)

**Low DRAM Mode (`--low_dram`)**:
- Images loaded on-demand, cached after first use
- Minimal RAM usage (~1-2GB for 185 images)
- ~5ms I/O penalty on first access M
- **Best for**: Limited RAM (<16GB), large datasets
- **Best for**: Most systems with 16GB+ RAM

**Default wothout:**:
- All images loaded at startup into RAM
- Faster training (less I/O overhead)
- Requires ~8GB RAM for 185 images at full resolution
- **Best for**: Systems with 32GB+ RAM, maximum speed

---

## Training Speed & Bottlenecks

### Expected Performance (RTX 5060 Ti, Garden dataset, --resolution 2)

| Phase | Iterations | Speed | GPU Util | VRAM | Duration |
|-------|-----------|-------|----------|------|----------|
| Startup | 0-500 | 2-4 it/s | 30-50% | 3-4 GB | 2-3 min |
| Densification | 500-15000 | 4-7 it/s | 40-90% | 6-10 GB | 35-50 min |
| Stable | 15000-30000 | 6-8 it/s | 85-100% | 8-12 GB | 30-35 min |
| **Total** | **30000** | **~7 it/s avg** | **~70% avg** | **peak 12 GB** | **~70 min** |

### Common Bottlenecks

1. **I/O (WSL /mnt/c)**
   - Reading from Windows filesystem → 20-30% slower
   - **Solution**: Copy dataset to native Linux path (e.g., `/home/user/datasets/`)

2. **Low GPU utilization (30-50%)**
   - **Cause**: Densification phase (CPU-bound operations)
   - **Normal**: Temporary drops during prune/split operations

3. **High VRAM usage (>14GB)**
   - **Cause**: Large scene, high resolution, excessive Gaussians
   - **Solution**: Use `--resolution 2` or `--densify_grad_threshold 0.0005`

4. **Slow convergence**
   - **Cause**: Learning rates too low/high
   - **Check**: Monitor loss curve in TensorBoard

---

## Example Commands

### Basic Training (Half Resolution)
```bash
conda activate sugar
cd mip-splatting/

python train.py \
  -s /path/to/colmap_dataset \
  -m /path/to/output \
  --resolution 2
```

### Custom Evaluation & Checkpoint Intervals
```bash
python train.py \
  -s /path/to/dataset \
  -m /path/to/output \
  --resolution 2 \
  --test_iterations 1000 5000 10000 15000 30000 \
  --checkpoint_iterations 5000 10000 20000 \
  --save_iterations 5000 10000 15000 20000 30000
```

**What this does:**
- **Evaluates** (L1/PSNR on test set) at: 1K, 5K, 10K, 15K, 30K iterations
- **Saves checkpoints** (resumable `.pth` files) at: 5K, 10K, 20K
- **Saves Gaussians** (final `.ply` models) at: 5K, 10K, 15K, 20K, 30K

### Target 6M Gaussians (VRAM-Safe Evaluation)
```bash
python train.py \
  -s ../SAMPLES/garden \
  -r 2 \
  --iteration 30000 \
  --target_gaussian_count 6000000 \
  --densify_until_iter 15000 \
  --min_opacity_threshold 0.01 \
  --eval \
  --test_iterations 15020 21000 25000 30000 \
  --save_iterations 16000 21000 25000 30000 \
  --checkpoint_iterations 8000 12000 16000 21000 26000 \
  --experiment_name "garden-r2-30k-6M-target"
```

**What this does:**
- **Binary search cap**: Enforces 6M Gaussian limit for 16GB VRAM safety
- **Tests at**: 15020 (post-densification), 21K, 25K, 30K (refinement checkpoints)
- **Saves at**: 16K (early refinement), 21K (mid), 25K (late), 30K (final)
- **Resume checkpoints**: Every 4-5K iterations for flexibility if VRAM issues occur

**Why test at 15020?** Testing right after densification (15000) with 6M Gaussians can spike VRAM during evaluation. Testing at 15020 ensures binary search cap is enforced first, keeping VRAM safe for 16GB GPUs.

### Low DRAM Mode (Minimal RAM Usage)
```bash
python train.py \
  -s /path/to/dataset \
  -m /path/to/output \
  --resolution 2 \
  --low_dram
```
**Use when:** You have limited RAM (<16GB) and need to save memory

### High Quality (Full Resolution)
```bash
python train.py \
  -s /path/to/dataset \
  -m /path/to/output \
  --resolution 1 \
  --iterations 40000
```

### Resume from Checkpoint
```bash
python train.py \
  -s /path/to/dataset \
  -m /path/to/output \
  --start_checkpoint /path/to/output/chkpnt15000.pth
```

### Enable Anti-Aliasing (Slower)
```bash
python train.py \
  -s /path/to/dataset \
  -m /path/to/output \
  --resolution 2 \
  --ray_jitter \
  --resample_gt_image
```

---

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir=/path/to/output --port 6006
```

**Key metrics:**
- `train_loss_patches/l1_loss` - L1 reconstruction loss
- `train_loss_patches/total_loss` - Combined L1 + SSIM loss
- `test/loss_viewpoint - psnr` - PSNR on test views
- `total_points` - Number of Gaussians
- `scene/opacity_histogram` - Opacity distribution

### Terminal Output
```
[ITER 7000] Evaluating test: L1 0.0234 PSNR 28.45
[ITER 7000] Saving Gaussians
[ITER 30000] Evaluating test: L1 0.0189 PSNR 29.87
Training complete.
```

---

## Advanced: Mip-Splatting Specifics

### 3D Gaussian Filtering (`kernel_size`)
- Mip-splatting adds **3D conical frustum filtering** to each Gaussian
- Computed once per camera view with `compute_3D_filter(cameras)`
- Updated after every densification
- `--kernel_size` controls filter radius (default 0.1)

### Ray Jittering (`--ray_jitter`)
- Adds random subpixel offset [-0.5, 0.5] to each ray
- Improves anti-aliasing but reduces training speed (~10% slower)
- Requires `--resample_gt_image` to offset ground truth images

### High-Resolution Sampling (`--sample_more_highres`)
- 30% chance to sample from cameras with width ≥ 800px
- Improves quality on high-res regions
- Useful for mixed-resolution datasets

---

## Troubleshooting

### "CUDA out of memory"
- Use `--resolution 2` or `--resolution 4`
- Ensure `--data_device cpu` (default) to keep images in RAM
- Reduce `--densify_until_iter 10000`

### "Slow training speed (<3 it/s)"
- Check if reading from Windows mount (`/mnt/c`) → copy to native Linux path
- Verify GPU isn't thermal throttling (`watch nvidia-smi`)
- Disable ray jittering if enabled

### "Floater artifacts in output"
- Increase `--densify_grad_threshold 0.0005` (less aggressive)
- Increase opacity prune threshold in code (0.005 → 0.01)

### "Training crashes after checkpoint load"
- Ensure `--start_checkpoint` path is absolute
- Check CUDA modules are compiled for your architecture

---

**For installation instructions, see:** [INSTALL.md](../INSTALL.md)  
**For SuGaR mesh extraction:** [SUGAR_USAGE.MD](SUGAR_USAGE.MD)
