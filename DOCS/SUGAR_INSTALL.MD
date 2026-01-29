# SuGaR Installation Summary

Quick reference for dependencies and compilation steps. **For complete installation guide, see:** [INSTALL.md](../INSTALL.md)

---

## ✅ Installation Overview

### Core Dependencies (via conda)
```bash
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
```
**Installs:**
- fvcore 0.1.5
- iopath 0.1.10
- yacs 0.1.8
- portalocker 3.2.0
- tabulate, termcolor, colorama

### Python Packages (via pip)
```bash
pip install PyMCubes
```
**Installs:**
- PyMCubes 0.1.6 (marching cubes mesh extraction)

### Compiled CUDA Modules (with compute capability optimizations)

#### 1. pytorch3d (v0.7.9)
```bash
cd pytorch3d/

# Set your GPU compute capability (example for Blackwell sm_120)
export TORCH_CUDA_ARCH_LIST="12.0"
export FORCE_CUDA=1
export MAX_JOBS=8

pip install --no-build-isolation -e .
```

**Find your compute capability:**
| GPU Series | Compute Capability |
|------------|-------------------|
| RTX 5000 (Blackwell) | 12.0 |
| RTX 4000 (Ada) | 8.9 |
| RTX 3000 (Ampere) | 8.6 |

**Optimizations:**
- Compiled for your GPU architecture
- TF32/FP16 support for Tensor Cores
- Used for: mesh operations, quaternions, rasterization, KNN, I/O

#### 2. nvdiffrast (v0.4.0)
```bash
cd nvdiffrast/

export TORCH_CUDA_ARCH_LIST="12.0"  # Set to your GPU
export MAX_JOBS=8

pip install --no-build-isolation .
```

**Optimizations:**
- Compiled for your GPU architecture
- Fast mesh rasterization for texture extraction
- Reduces texture baking time from minutes to ~10 seconds

---

## Key Insight: No Gaussian Rasterization Compilation Needed!

**SuGaR can use mip-splatting output directly via `--gs_output_dir`**

This means:
- ✅ No need to compile `diff-gaussian-rasterization` in SuGaR
- ✅ No need to compile `simple-knn` in SuGaR  
- ✅ SuGaR just loads the trained Gaussians from mip-splatting
- ✅ Saves compilation time and avoids potential conflicts

**Workflow:**
1. Train with mip-splatting (7k-30k iterations)
2. Pass output to SuGaR with `--gs_output_dir`
3. SuGaR applies surface alignment and extracts mesh
4. Export to Blender for editing/animation

---

## Verification

### Test PyTorch & GPU
```bash
conda run -n sugar python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Capability: {torch.cuda.get_device_capability(0)}')
"
```
**Expected output:**
```
PyTorch: 2.x.x
CUDA: 13.0 (or 12.4)
GPU: NVIDIA GeForce RTX 5060 Ti (or your GPU)
Capability: (12, 0) (or your compute capability)
```

### Test pytorch3d
```bash
python -c "import pytorch3d; print(f'pytorch3d: {pytorch3d.__version__}')"
```
**Expected:** `pytorch3d: 0.7.9`

### Test nvdiffrast
```bash
python -c "import nvdiffrast.torch as dr; print('nvdiffrast: OK')"
```

---

## Example Usage

### Train SuGaR from Mip-Splatting Output
```bash
conda activate sugar
cd SuGaR/

python train_full_pipeline.py \
  -s /path/to/dataset \
  --gs_output_dir /path/to/mip_output \
  -r dn_consistency \
  --high_poly True \
  --export_obj True \
  --refinement_time short
```

**This will:**
1. Load trained Gaussians from mip-splatting output
2. Apply dn_consistency regularization for surface alignment
3. Extract 1M vertex mesh with 1 Gaussian per triangle
4. Refine for 2k iterations (~5 minutes)
5. Export textured .obj mesh for Blender

**Output location:**
```
output/refined_dn_consistency/<scene_name>/
├── point_cloud/point_cloud.ply              # Refined Gaussians
├── refined_mesh/mesh.ply                    # Mesh (no texture)
├── refined_mesh_with_texture/mesh.obj       # For Blender
└── refined_ply/gaussians.ply                # For viewers
```

---

## Documentation

Complete guides available:

1. ✅ **[SUGAR_USAGE.MD](SUGAR_USAGE.MD)** - Complete SuGaR usage guide
   - Parameter reference
   - Pipeline stages
   - Troubleshooting
   - Optimization tips

2. ✅ **[SUGAR_OPTIMISATIONS.MD](SUGAR_OPTIMISATIONS.MD)** - Performance optimizations
   - 12 key optimizations explained
   - VRAM reduction strategies
   - Speed improvements
   - Implementation details

3. ✅ **[MIPS_TRAIN.MD](MIPS_TRAIN.MD)** - Mip-Splatting training
   - Training parameters
   - Phase breakdown
   - Performance benchmarks

---

## Next Steps

### Optional Tools

**SuperSplat Viewer:**
SuperSplat is NOT included in this repository.
- **Web Viewer (Recommended):** https://playcanvas.com/supersplat/editor - No installation, drag & drop your `.ply` files
- **Local Install:** https://github.com/playcanvas/supersplat - Requires Node.js 18+

**Blender Integration:**
Install Blender add-on from: https://github.com/Anttwo/sugar_frosting_blender_addon/

### Test Run
Try SuGaR on a small dataset to verify everything works:
```bash
cd SuGaR/
python train_full_pipeline.py \
  -s /path/to/test_scene \
  --gs_output_dir /path/to/mip_output \
  -r dn_consistency \
  --low_poly True \
  --refinement_time short
```

---

## Compilation Flags Reference

All CUDA modules should be compiled with:
- `TORCH_CUDA_ARCH_LIST="X.Y"` - Target your GPU architecture
- `FORCE_CUDA=1` - Force CUDA compilation
- `MAX_JOBS=8` - Parallel compilation
- `--no-build-isolation` - Use environment PyTorch

This ensures optimal performance with Tensor Core acceleration (TF32, FP16, BF16).

---

**For complete installation guide:** [INSTALL.md](../INSTALL.md)  
**For usage examples:** [SUGAR_USAGE.MD](SUGAR_USAGE.MD)  
**For optimizations:** [SUGAR_OPTIMISATIONS.MD](SUGAR_OPTIMISATIONS.MD)
