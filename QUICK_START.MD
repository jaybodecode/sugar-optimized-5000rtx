# SugarV3 Quick Start

**Project:** Mip-Splatting → SuGaR pipeline for 3D reconstruction  
**Target:** RTX 5060 Ti (16GB VRAM), optimized for VRAM efficiency

**🎯 NEW: Defaults Optimized for RTX 5060 Ti 16GB VRAM**
- **Binary search hard cap:** 5M Gaussians (prevents VRAM overflow)
- **Emergency brake:** Densification stops at 8k iterations, aggressive 0.03 opacity pruning
- **Safe evaluation:** eval_camera_stride=3 (every 3rd camera)
- **24GB+ GPUs:** Override with `--target_gaussian_count 0 --densify_until_iter 15000 --min_opacity_threshold 0.005 --eval_camera_stride 1`

**Note:** All paths in this guide use relative paths from the cloned repository root.  
The sample dataset is located at `SAMPLES/garden` and outputs will be saved to `SAMPLES/garden_output/<timestamp>`.

**📦 Sample Dataset:** If using the included Garden scene, make sure you've cloned the SAMPLES submodule during installation (see [INSTALL.md](INSTALL.md) Step 6).

---

## 1. Activate Environment

```bash
conda activate sugar
```

## 2. Mip-Splatting Training (Gaussian Generation)

```bash
cd mip-splatting

# For 16GB VRAM (RTX 5060 Ti, 5070, 5080) - Defaults optimized for 16GB VRAM
python train.py \
  -s ../SAMPLES/garden \
  --iteration 30000 \
  --test_iterations 1000 5000 7000 15000 30000 \
  --checkpoint_iterations 7000 15000 30000 \
  --save_iterations 7000 15000 30000 \
  -r 2 \
  --experiment_name "garden-r2-30k"

# Re-run without prompt (auto-deletes existing output)
python train.py \
  -s ../SAMPLES/garden \
  --iteration 30000 \
  -r 2 \
  --experiment_name "garden-r2-30k" \
  --delete_first

# For 16GB VRAM - Override defaults (disable protections for 24GB+ GPU)
python train.py \
  -s ../SAMPLES/garden \
  --iteration 30000 \
  -r 2 \
  --target_gaussian_count 0 \
  --eval_camera_stride 1 \
  --densify_until_iter 15000 \
  --min_opacity_threshold 0.005 \
  --experiment_name "garden-no-limits"

# For 32GB VRAM (RTX 5090) - Full quality, 30K iterations with phase-aligned checkpoints
# Note: -r 1 (full resolution) requires 19-21GB VRAM for mip-splatting
# For 16GB VRAM: Use -r 2 (half-res) instead, or see SuGaR optimizations below
python train.py \
  -s ../SAMPLES/garden \
  --iteration 30000 \
  --test_iterations 1000 2500 5000 7000 10000 15000 20000 25000 30000 \
  --checkpoint_iterations 2500 7000 15000 22000 30000 \
  --save_iterations 7000 15000 30000 \
  -r 1 \
  --target_gaussian_count 0 \
  --eval_camera_stride 1 \
  --densify_until_iter 15000 \
  --min_opacity_threshold 0.005 \
  --experiment_name "garden-r1-30k"

# For 16GB VRAM - Maximum Quality Strategy (Two-stage approach):
# Stage 1: Train mip-splatting at -r 2 (fits in 16GB)
# Stage 2: SuGaR training with Phase 1 optimizations (see below)
# Result: High-quality mesh from -r 2 Gaussians (still excellent for Unity)
python train.py \
  -s ../SAMPLES/garden \
  --iteration 40000 \
  -r 2 \
  --target_gaussian_count 6000000 \
  --densify_until_iter 28000 \
  --eval \
  --test_iterations 7000 15000 28000 36000 40000 \
  --save_iterations 28000 40000 \
  --checkpoint_iterations 28000 36000 \
  --experiment_name "garden-r2-40k-for-unity"

# Quality-optimized: 6M Gaussians, 40K iterations (optimal stopping point)
# Recommended for maximum quality within 16GB VRAM constraints
python train.py \
  -s ../SAMPLES/garden \
  -r 2 \
  --iteration 40000 \
  --target_gaussian_count 6000000 \
  --densify_until_iter 28000 \
  --densify_grad_threshold 0.00012 \
  --min_opacity_threshold 0.02 \
  --eval \
  --test_iterations 15000 20000 25000 28000 32000 36000 40000 \
  --save_iterations 28000 36000 40000 \
  --checkpoint_iterations 20000 28000 32000 36000 \
  --experiment_name "garden-r2-40k-6M-optimal"
```

**Mip-Splatting Training Phases (see [DOCS/MIPS_USAGE.MD](DOCS/MIPS_USAGE.MD) § Training Phases):**
- **Phase 1: Initialization (0-500)** - Load COLMAP data, initialize Gaussians (2-4 it/s warmup)
- **Phase 2: Densification (500-15000)** - Split/clone/prune every 100 iters, Gaussians grow 50K→5M (4-7 it/s, fluctuates)
  - Opacity reset at 3000 (prevents floaters)
- **Phase 3: Stable Training (15000+)** - Refinement only, no densification, Gaussian count stable (6-8 it/s)

**Test/Checkpoint Strategy:**
- **Test before phase changes:** `2500` (before pruning), `15000` (end of densification)
- **Checkpoints at phase boundaries:** `2500` (pre-pruning resume point), `7000` (SuGaR-compatible), `15000` (critical milestone)
- **Save iterations:** `7000, 15000, 30000` (key milestones only to save disk space)

**Fast Evaluation (--eval_camera_stride):**
- `--eval_camera_stride 1` (default) - Full evaluation, all test cameras
- `--eval_camera_stride 2` - Evaluates every 2nd camera (50% faster, good for 16GB VRAM)
- ⚠️ **If VRAM issues during eval:** Use stride 2-4, see [DOCS/MIPS_USAGE.MD](DOCS/MIPS_USAGE.MD) for tuning

**Data Device (--data_device):**
- `--data_device cuda` - Copies all source images (uncompressed) into VRAM for faster training
- ⚠️ **VRAM cost:** At -r1 resolution, Garden dataset (~20MP images) requires ~10-15GB just for images
- **Use with 32GB VRAM at -r2 or lower** (RTX 5090 recommended)
- **Not recommended for -r1 full resolution** - will likely OOM even with 32GB
- For lower VRAM systems or -r1 training, omit this flag (uses CPU by default)

**Memory Management (Lazy Loading with LRU Cache):**
- **Default:** Lazy loading with LRU cache of 2GB
- Images load on-demand and cache evicts oldest when full
- **Low RAM systems:** Use `--image_cache_gb 1.0` for minimal RAM usage (see [DOCS/MIPS_USAGE.MD](DOCS/MIPS_USAGE.MD))
- **High RAM systems:** Use `--image_cache_gb 5.0` for faster training

**Example with high resolution and tuned cache:**
```bash
python train.py \
  -s ../SAMPLES/garden \
  --iteration 7000 \
  -r 1 \
  --image_cache_size 10 \
  --experiment_name "garden-r1-7k-lowram"
```

**Parameters:**
- `-s` - Scene dataset path
- `-r N` - Resolution downsample factor (lower = higher quality, slower)
  - **Source images:** ~20MP (5187×3361 pixels)
  - `-r 4`: ~1.3MP (~15-20 min for 7K iter) - Fast testing
  - `-r 2`: ~5MP (~30 min for 7K iter) - **Required for 16GB VRAM** (RTX 5060 Ti, 5070, 5080) ✅
  - `-r 1`: ~20MP (~2.5 hours for 7K iter, ~8 hours for 30K iter) - **Requires 32GB VRAM** (RTX 5090 or higher)
- `--iteration N` - Total training iterations
- `--data_device cuda` - Keep images on GPU (32GB VRAM only)
- See [DOCS/MIPS_USAGE.MD](DOCS/MIPS_USAGE.MD) for complete parameter reference

**⚠️ VRAM Requirements for Sample Dataset (Garden @ ~20MP):**
- **16GB VRAM (5060 Ti / 5070 / 5080):** Use `-r 2` (half-res) with default settings
- **32GB VRAM (5090):** Can use `-r 1` (full-res) for maximum quality

**Monitor:**
```bash
tensorboard --logdir "../SAMPLES/garden_output" --port 6006 --bind_all
```

**Resume from Checkpoint:**
```bash
python train.py \
  -s ../SAMPLES/garden \
  --iteration 30000 \
  -r 2 \
  --start_checkpoint ../SAMPLES/garden_output/garden-r2-7k/chkpnt7000.pth \
  --experiment_name "garden-r2-30k-resumed"
```

**⚠️ Checkpoint Safety - Safe Parameter Changes:**
When resuming from checkpoints, only these parameters are safe to change:
- ✅ **Learning rates** - optimizer will adapt
- ✅ **Test/save iterations** - just scheduling
- ✅ **Resolution** (-r flag) - just affects rendering
- ✅ **Evaluation settings** (--eval_camera_stride, --enable_lpips)
- ✅ **Output paths** (--experiment_name, --model_path)

**⚠️ Checkpoint Contamination Risk:**
Changing these parameters mid-training causes optimizer conflicts and VRAM explosions:
- ❌ **Densification parameters** (--densify_grad_threshold, --densify_until_iter, --min_opacity_threshold)
- ❌ **Opacity reset interval**
- ❌ **SH degree** (changes model structure)

**If you need to change growth/pruning parameters, start fresh training without --start_checkpoint!**

---

## 3. SuGaR Full Pipeline (Mesh + Texture)

### Option A: Full Pipeline (Recommended for first run)

**For 16GB VRAM (RTX 5060 Ti, 5070, 5080):**
```bash
cd SuGaR

python train_full_pipeline.py \
  -s ../SAMPLES/garden \
  --gs_output_dir ../SAMPLES/garden_output/<timestamp> \
  -r density \
  --high_poly True \
  --refinement_time long \
  --eval True
```

**For 24GB+ VRAM (RTX 5090, A5000+):**
```bash
cd SuGaR

python train_full_pipeline.py \
  -s ../SAMPLES/garden \
  --gs_output_dir ../SAMPLES/garden_output/<timestamp> \
  -r dn_consistency \
  --high_poly True \
  --refinement_time long \
  --eval True
```

### Option B: Training Only (More control, can resume)

**For 16GB VRAM (RTX 5060 Ti, 5070, 5080):**
```bash
cd SuGaR

python train.py \
  -s ../SAMPLES/garden \
  -c ../SAMPLES/garden_output/<timestamp> \
  -r density \
  --high_poly True \
  --refinement_time long \
  --eval True
```

**For 24GB+ VRAM (RTX 5090, A5000+):**
```bash
cd SuGaR

python train.py \
  -s ../SAMPLES/garden \
  -c ../SAMPLES/garden_output/<timestamp> \
  -r dn_consistency \
  --high_poly True \
  --refinement_time long \
  --eval True
```

**Note:** Use `-c` (checkpoint_path) in `train.py`, but `--gs_output_dir` in `train_full_pipeline.py`

**Key Parameters:**
- `--gs_output_dir` - Path to mip-splatting output (use actual timestamp folder)
- `-r density` - **Recommended for 16GB VRAM** (73% VRAM usage, 1.0-1.2 it/s)
- `-r dn_consistency` - Best quality, requires 24GB+ VRAM (depth-normal consistency)
- `--high_poly True` - High-poly mesh extraction
- `--refinement_time long` - 7K iterations (~2-3 hours with optimizations)
- See [DOCS/SUGAR_USAGE.MD](DOCS/SUGAR_USAGE.MD) for full parameter list

**🚀 Phase 1 VRAM Optimizations (Applied in v2 - Reduces peak VRAM by 4-6 GB):**

SuGaR training now includes these optimizations for better VRAM efficiency:
- ✅ **Reduced SDF samples:** 250K (was 1M) - Saves ~4 GB VRAM
- ✅ **Progressive resolution warmup:** Enabled - Saves ~6 GB in early training
- ✅ **Tensor cleanup:** Explicit cleanup of entropy, depth-normal, test tensors - Saves ~0.5-1 GB

**Expected VRAM usage with optimizations:**
- Previous: ~15.7 GB peak
- Now: **~11-12 GB peak** (4-6 GB reduction)
- Enables future `-r 1` mip-splatting outputs with 16GB VRAM

**SuGaR Fine-Tuning Strategies (From 40K Mip-Splatting Checkpoint):**

**📊 Iteration Selection Guide:**

| Strategy | Coarse Iterations | Time | Quality | Use Case |
|----------|------------------|------|---------|----------|
| **Quick** | 40K → 45K (5K) | ~1.5h | Good | Fast iteration/testing |
| **Optimal** | 40K → 50K (10K) | ~3h | Excellent | **Recommended for production** |
| **Conservative** | 40K → 55K (15K) | ~4.5h | Excellent+ | When 50K needs validation |
| **Overkill** | 40K → 65K (25K) | ~7-8h | Marginal gains | Diminishing returns |

**💡 Why 50K is Optimal:**
- 40K checkpoint already well-trained (PSNR ~27-30)
- 10K fine-tuning sufficient for SDF regularization
- Mesh quality comparable to 65K
- Saves 4-5 hours vs 65K

**Example: Optimal Strategy (40K → 50K)**

**For 16GB VRAM (RTX 5060 Ti):**
```bash
cd SuGaR

# Full pipeline (coarse + mesh + refine + texture)
nohup python train.py \
  -s ../SAMPLES/garden \
  -c ../SAMPLES/garden_output/garden-r2-60k-6M-quality \
  -i 40000 \
  -r density \
  --high_poly True \
  --refinement_time long \
  --experiment_name "6M-quality-5Mverts-50K" \
  --coarse_iterations 50000 \
  --checkpoint_interval 2000 \
  --checkpoint_milestones 45000 48000 50000 \
  --test_iterations 48000 50000 \
  --export_ply True \
  --eval True \
  --delete_first \
  > garden-mesh-50K.log 2>&1 &

# Monitor progress
tail -f garden-mesh-50K.log
```

**Performance (RTX 5060 Ti 16GB):**
- VRAM: 73% (~11.7 GB)
- Speed: 1.0-1.2 it/s
- Time: ~2-3 hours for 10K iterations (40K→50K)

**For 24GB+ VRAM (better quality):**
```bash
# Use -r dn_consistency instead for depth-normal consistency
-r dn_consistency
```

**🎯 Parameter Adjustment Guide:**

**For 45K (Quick):**
```bash
--coarse_iterations 45000
--checkpoint_milestones 43000 45000
--test_iterations 45000
```

**For 50K (Optimal - Recommended):**
```bash
--coarse_iterations 50000
--checkpoint_milestones 45000 48000 50000
--test_iterations 48000 50000
```

**For 55K (Conservative):**
```bash
--coarse_iterations 55000
--checkpoint_milestones 48000 52000 55000
--test_iterations 52000 55000
```

**Total Pipeline Time (40K checkpoint → Final mesh):**
- Coarse (40K→50K): ~3h
- Mesh extraction: ~30-60min
- Refinement (15K iter): ~3-4h
- Texture export: ~10min
- **Total: ~7-8 hours** (fits overnight run)

**Optimized Performance (RTX 5060 Ti 16GB):**
- VRAM usage: ~11-12 GB peak (was ~15.7 GB)
- Speed: 1.2-2.8 it/s (~2-3 min per 200 iterations)
  - Varies with neighbor resets and memory pressure
  - 15K iterations: ~2-2.5 hours total
- Initialization: ~5 seconds (with cache)
- See [DOCS/SUGAR_OPTIMISATIONS.MD](DOCS/SUGAR_OPTIMISATIONS.MD) for optimization details

**Monitor:**
```bash
tensorboard --logdir ./output/coarse/<scene>/sugarcoarse_*/tensorboard --port 6007 --bind_all
```

---

## 4. Continue to Refinement Stage (After Coarse Training Complete)

**When coarse training finishes (15000 iterations), continue to refinement:**

```bash
cd SuGaR

python train.py \
  -s ../SAMPLES/garden \
  -c ../SAMPLES/garden_output/<timestamp> \
  -r dn_consistency \
  --high_poly True \
  --refinement_time long \
  --eval True \
  --resume_checkpoint output/coarse/garden/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt
```

**⚠️ IMPORTANT:** Always use `--resume_checkpoint` pointing to the final checkpoint (15000.pt) to:
- ✅ Skip coarse training (already complete)
- ✅ Extract coarse mesh from trained model
- ✅ Start refinement stage (7000 iterations with "long")
- ✅ Prevent accidental restart from iteration 0

**Without `--resume_checkpoint`:** Training will restart from scratch! ❌

---

## 5. Resume Training (If Interrupted During Coarse Stage)

**Resume from any saved checkpoint (if interrupted mid-training):**

```bash
cd SuGaR

python train.py \
  -s ../SAMPLES/garden \
  -c ../SAMPLES/garden_output/<timestamp> \
  -r dn_consistency \
  --high_poly True \
  --refinement_time long \
  --eval True \
  --resume_checkpoint output/coarse/garden/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/14000.pt
```

**Available checkpoints (every 1000 iterations):**
```bash
# Resume from different points
--resume_checkpoint .../7000.pt   # Resume from iteration 7000
--resume_checkpoint .../10000.pt  # Resume from iteration 10000
--resume_checkpoint .../14000.pt  # Resume from iteration 14000
--resume_checkpoint .../15000.pt  # Coarse complete, start refinement
```

**What gets restored:**
- ✅ Model weights (Gaussians, opacity, scales, etc.)
- ✅ Optimizer state (learning rates, momentum)
- ✅ Training iteration count
- ✅ Loss history
- ✅ TensorBoard logs continue in same directory

---

## 6. Output Files

**Mip-splatting output:**
- `point_cloud/iteration_7000/point_cloud.ply` - Trained Gaussians
- Checkpoints in `chkpnt7000.pth`

**SuGaR output:**
- `output/coarse/<scene>/sugarcoarse_*/` - Coarse training checkpoints
  - `1000.pt`, `2000.pt`, ... `15000.pt` - Training checkpoints (can resume from any)
  - `tensorboard/` - TensorBoard logs
- `output/refined/<scene>/` - Refined mesh with texture
- `textured_mesh.obj` + `material_0.png` - Final textured mesh
- See [DOCS/OUTPUT_USAGE/](DOCS/OUTPUT_USAGE/) for viewing instructions

---

## 7. Quick References

- **Installation:** [DOCS/SUGAR_INSTALL.MD](DOCS/SUGAR_INSTALL.MD)
- **Mip-Splatting Guide:** [DOCS/MIPS_TRAIN.MD](DOCS/MIPS_TRAIN.MD)  
- **SuGaR Usage:** [DOCS/SUGAR_USAGE.MD](DOCS/SUGAR_USAGE.MD)
- **Optimizations:** [DOCS/SUGAR_OPTIMISATIONS.MD](DOCS/SUGAR_OPTIMISATIONS.MD)
- **Project Status:** [LLM.MD](LLM.MD) - Current tasks and workflow
- **Future Work:** [TODO.MD](TODO.MD) - Planned features
