# Installation Guide - SuGaR v3

Complete installation instructions for the **Mip-Splatting → SuGaR → Textured Mesh** pipeline optimized for NVIDIA RTX 5000 Series (Blackwell architecture).

---

## � Step 1: CUDA Toolkit Installation (REQUIRED FIRST)

**⚠️ Install CUDA 13.0+ BEFORE running install.sh**

CUDA toolkit is required to compile CUDA extensions (pytorch3d, nvdiffrast, Gaussian Splatting modules).

### WSL2 (Windows):

```bash
# Download CUDA keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb

# Install keyring
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# Update and install CUDA toolkit 13.0
sudo apt-get update
sudo apt-get install -y cuda-toolkit-13-0

# Add to PATH (add to ~/.bashrc for persistence)
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

# Verify installation
nvcc --version  # Should show "release 13.0"
```

### Native Linux:

Download from: [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
- Select: Linux → x86_64 → Ubuntu → 22.04 → deb (network)
- Follow on-screen instructions
- Choose CUDA 13.0 or newer

### Verify CUDA Installation:

```bash
nvcc --version
# Expected: Cuda compilation tools, release 13.0 or newer
```

**Do not proceed until `nvcc --version` works!**

---

## 💡 Step 2: WSL2 Setup (Windows Only)

**This project supports Linux and WSL2 only (no native Windows).**

**If you're on Windows:**

1. **Install WSL2** with Ubuntu 22.04:
   ```powershell
   wsl --install -d Ubuntu-22.04
   ```

2. **Install NVIDIA drivers 570+ on Windows** (not in WSL):
   - Download: [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
   - Select: GeForce RTX 5000 Series, Windows 11, Game Ready Driver
   - Version 570.00 or newer required for Blackwell (RTX 5060 Ti)

3. **Verify GPU access in WSL2:**
   ```bash
   nvidia-smi  # Should show your RTX 5060 Ti and driver 570+
   ```

4. **Configure WSL2 Resources (Recommended):**
   
   Create or edit `C:\Users\<YourUsername>\.wslconfig` (on Windows side) to optimize performance:
   
   ```ini
   [wsl2]
   memory=24GB       # Limits the WSL2 VM to use no more than 24 GB of RAM. Based on 32GB Windows system (sufficient for ~20MP high-res training)
   processors=10     # Assign 10 virtual processors (assuming you have 12+ cores)
   swap=16GB         # Amount of swap space (default is 25% of RAM)
   swapfile=C:\\temp\\wsl-swap.vhdx  # Optional: custom swap file location. Shared by all WSL2 instances. Use fastest drive!
   localhostForwarding=true
   ```
   
   **Apply changes:**
   ```powershell
   # In Windows PowerShell/CMD:
   wsl --shutdown
   # Then restart WSL by opening Ubuntu
   ```
   
   **Important:** All WSL2 distros share the same VM and swap file. These limits apply to ALL distros combined.

---

## 🚀 Step 3: Automated Installation

**One-command automated installer** (recommended):

```bash
git clone https://github.com/jaybodecode/sugar-optimized-5000rtx.git
cd sugar-optimized-5000rtx
bash install.sh
```

**Or non-interactive mode:**
```bash
bash install.sh -y
```

This will:
- Check prerequisites (CUDA toolkit, conda, GPU access)
- Create conda environment with Python 3.12
- Install PyTorch with CUDA 13.0 support
- Install all Python dependencies
- Compile pytorch3d, nvdiffrast, and Gaussian Splatting CUDA modules
- Run comprehensive verification tests

**One-command automated installer** (recommended):

```bash
git clone https://github.com/jaybodecode/sugar-optimized-5000rtx.git
cd sugar-optimized-5000rtx
bash install.sh
```

This will:
- Check prerequisites (conda/mamba, GPU access)
- Create conda environment with Python 3.12
- Install PyTorch with CUDA support (uses Windows CUDA on WSL2)
- Install all Python dependencies
- Compile pytorch3d, nvdiffrast, and Gaussian Splatting CUDA modules
- Run comprehensive verification tests

**Manual installation:** Follow the detailed steps below.

---

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Core Dependencies](#core-dependencies)
- [CUDA Compilation](#cuda-compilation)
- [Verification](#verification)
- [Optional Tools](#optional-tools)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

**Tested Configuration:**
- NVIDIA RTX 5060 Ti (16GB VRAM) - ✅ Fully tested and optimized
- 32GB System RAM
- 50GB free disk space

**RTX 5000 Series Only:**
- **RTX 5060 Ti (16GB):** ✅ Fully tested and optimized
- **RTX 5070 (12GB):** Should work (untested)
- **RTX 5080 (16GB+):** Should work (untested)
- **RTX 5090 (32GB):** For 48MP images (untested)

**⚠️ RTX 4000/3000 series NOT SUPPORTED** - This project uses Blackwell-specific optimizations (sm_120) that require RTX 5000 series GPUs. The install script will block installation on older GPUs.

**Recommended System:**
- 64GB System RAM
- NVMe SSD with 100GB+ free space

### Software Requirements

- **OS:** Linux (Ubuntu 22.04+ recommended) **OR** WSL2 on Windows 11
- **CUDA:** 12.4+ (13.x for Blackwell GPUs) - **Native Linux only, NOT needed for WSL2**
- **Python:** 3.10, 3.11, or 3.12
- **Git:** For cloning repositories
- **Conda/Mamba:** For environment management

**Note:** Native Windows is NOT supported. Windows users must use WSL2.

---

## CUDA Installation

### Check Your Current CUDA Version

Before proceeding, verify your CUDA installation:

```bash
# Check CUDA compiler version
nvcc --version

# Check NVIDIA driver and CUDA runtime
nvidia-smi
```

**Expected output:**
- `nvcc --version` should show **CUDA 13.0+** for Blackwell GPUs (RTX 5060 Ti)
- `nvidia-smi` should show **Driver Version: 570.00+** and **CUDA Version: 13.0+**

### Do You Need to Install CUDA?

**You need CUDA 13.x if:**
- You have an RTX 5000 series GPU (Blackwell architecture)
- Your `nvcc --version` shows CUDA 12.x or earlier
- Your `nvidia-smi` shows Driver < 570

**CUDA 12.4+ is acceptable if:**
- You have an RTX 4000 series or older GPU
- You're not using Blackwell-specific optimizations

### Installing CUDA 13.x (Blackwell GPUs)

**Ubuntu 22.04 / 24.04:**

```bash
# 1. Remove old CUDA installations (optional but recommended)
sudo apt-get --purge remove "*cuda*" "*cublas*" "*cufft*" "*cufile*" "*curand*" "*cusolver*" "*cusparse*" "*gds-tools*" "*npp*" "*nvjpeg*" "nsight*" "*nvvm*"
sudo apt-get autoremove

# 2. Download and install CUDA 13.0
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-0

# 3. Install NVIDIA Driver 570+ (if not already installed)
sudo apt-get install -y nvidia-driver-570

# 4. Add CUDA to PATH
echo 'export PATH=/usr/local/cuda-13.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 5. Reboot to load new driver
sudo reboot
```

**After reboot, verify:**
```bash
nvcc --version  # Should show CUDA 13.0
nvidia-smi      # Should show Driver 570+ and CUDA 13.0
```

### Troubleshooting CUDA Installation

**Issue: "nvcc: command not found" after installation**
```bash
# Manually add CUDA to current session
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Issue: nvidia-smi shows CUDA but nvcc is missing**
- NVIDIA driver provides runtime CUDA (shown in `nvidia-smi`)
- You still need CUDA Toolkit for compilation: `sudo apt-get install cuda-toolkit-13-0`

**Issue: Multiple CUDA versions installed**
```bash
# List installed versions
ls /usr/local/ | grep cuda

# Set specific version
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
```

---

## Environment Setup

### 1. Install Conda

If you don't have conda:

```bash
# Install Miniconda (recommended)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Follow prompts, then restart your shell or run:
source ~/.bashrc
```

**Note:** The install.sh script can auto-install Miniconda if you don't have it.

### 2. Clone Repository

```bash
git clone https://github.com/jaybodecode/sugar-optimized-5000rtx.git
cd sugar-optimized-5000rtx
```

### 3. Create Conda Environment

```bash
# Create environment with Python 3.12
conda create -n sugar python=3.12 -y
conda activate sugar

# Install PyTorch with CUDA support
# For CUDA 13.x (Blackwell/RTX 5000 series)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

# For CUDA 12.4 (Ada Lovelace/RTX 4000 series)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Verify PyTorch:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Core Dependencies

### 1. Install via Conda

```bash
conda activate sugar
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
```

This installs:
- `fvcore` - Facebook core utilities
- `iopath` - I/O path management
- `yacs` - Configuration system
- `portalocker` - File locking
- `tabulate`, `termcolor`, `colorama` - Terminal formatting

### 2. Install Python Packages

```bash
pip install -r SuGaR/requirements.txt
```

**Key packages:**
- `tqdm` - Progress bars
- `plyfile` - PLY file I/O
- `tensorboard` - Training visualization
- `rich` - Rich terminal output
- `opencv-python` - Image processing
- `scikit-image` - Image utilities
- `PyMCubes` - Marching cubes (mesh extraction)
- `huggingface_hub` - Model downloads

---

## CUDA Compilation

**Important:** Set your GPU's compute capability. Find yours [here](https://developer.nvidia.com/cuda-gpus).

| GPU Architecture | Compute Capability | TORCH_CUDA_ARCH_LIST |
|------------------|-------------------|----------------------|
| Blackwell (RTX 5000 series) | 12.0 | "12.0" |
| Ada Lovelace (RTX 4000 series) | 8.9 | "8.9" |
| Ampere (RTX 3000 series) | 8.6 | "8.6" |
| Turing (RTX 2000 series) | 7.5 | "7.5" |

### 1. pytorch3d (Mesh Operations & KNN)

```bash
cd pytorch3d

# Set your compute capability (example for Blackwell)
export TORCH_CUDA_ARCH_LIST="12.0"
export FORCE_CUDA=1
export MAX_JOBS=8

# Compile and install
pip install --no-build-isolation -e .
```

**Verify:**
```bash
python -c "import pytorch3d; print(f'pytorch3d: {pytorch3d.__version__}')"
```

### 2. nvdiffrast (GPU Rasterization for Textures)

```bash
cd ../nvdiffrast

export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=8

# Compile and install
pip install --no-build-isolation .
```

**Verify:**
```bash
python -c "import nvdiffrast.torch as dr; print('nvdiffrast: OK')"
```

---

## Verification

### Complete System Check

Create a test script:

```python
# test_install.py
import sys
print(f"Python: {sys.version.split()[0]}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")

import pytorch3d
print(f"pytorch3d: {pytorch3d.__version__}")

import nvdiffrast.torch as dr
print("nvdiffrast: OK")

from rich import print as rprint
rprint("[bold green]✓ All dependencies installed successfully![/bold green]")
```

Run:
```bash
conda activate sugar
python test_install.py
```

**Expected output:**
```
Python: 3.11.x
PyTorch: 2.x.x
CUDA Available: True
CUDA Version: 13.0 (or 12.4)
GPU: NVIDIA GeForce RTX 5060 Ti
VRAM: 16.0 GB
Compute Capability: (12, 0)
pytorch3d: 0.7.9
nvdiffrast: OK
✓ All dependencies installed successfully!
```

---

## Optional Tools

### SuperSplat (3D Gaussian Viewer)

**Note:** SuperSplat is NOT included in this repository.

**Option 1: Web Viewer (Recommended)**
- Use the free online viewer: https://playcanvas.com/supersplat/editor
- No installation required
- Works directly in your browser
- Just drag and drop your `.ply` files

**Option 2: Local Installation**
- If you prefer to run locally: https://github.com/playcanvas/supersplat
- Requires Node.js 18+
- Follow their installation instructions

### Blender Integration

For exporting meshes to Blender:

1. Install [Blender 4.0+](https://www.blender.org/download/)
2. Install [SuGaR Frosting Blender Add-on](https://github.com/Anttwo/sugar_frosting_blender_addon/)

See [DOCS/SUGAR.MD](DOCS/SUGAR.MD) for workflow details.

---

## Troubleshooting

### CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Reduce batch size in training config
2. Enable gradient checkpointing
3. Use `--low_poly True` for SuGaR extraction
4. Close other GPU applications

### Compilation Errors

**Error:** `error: identifier "xxxxx" is undefined`

**Solutions:**
1. Verify CUDA toolkit version matches PyTorch:
   ```bash
   nvcc --version
   python -c "import torch; print(torch.version.cuda)"
   ```
2. Set correct `TORCH_CUDA_ARCH_LIST` for your GPU
3. Ensure GCC version is compatible (GCC 11 recommended):
   ```bash
   gcc --version
   ```

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'fvcore'`

**Solution:**
```bash
conda activate sugar
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
```

### TensorBoard Issues

**Error:** TensorBoard not showing training curves

**Solution:**
```bash
# In a separate terminal
conda activate sugar
tensorboard --logdir=./SuGaR/output
```

Access at: http://localhost:6006

### nvdiffrast "Invalid device context" Error

**Error:** `RuntimeError: Invalid device context`

**Solution:**
Ensure CUDA context is initialized before using nvdiffrast:
```python
import torch
torch.cuda.init()  # Initialize CUDA
import nvdiffrast.torch as dr
```

---

## 📦 Step 6: Download Sample Dataset (Optional)

Sample datasets are **not included** in this repository due to GitHub size limits. Download them separately:

```bash
# Download Mip-NeRF 360 dataset (18GB)
wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
unzip 360_v2.zip

# Extract to SAMPLES directory
mkdir -p SAMPLES
mv 360_v2/bicycle SAMPLES/
mv 360_v2/garden SAMPLES/
```

See [SAMPLES.md](SAMPLES.md) for detailed instructions. If you have your own dataset, skip this step and point the `-s` parameter to your dataset location.

**Recommended sample scenes:**
- Garden scene from Mip-NeRF 360 dataset
- 161 images at ~20MP resolution (5187×3361)
- COLMAP camera poses and sparse reconstruction
- Output directory structure examples

---

## Quick Start

Once installation is complete, see:
- [QUICK_START.MD](QUICK_START.MD) - Get running in 5 minutes
- [README.md](README.md) - Project overview and features
- [DOCS/SUGAR.MD](DOCS/SUGAR.MD) - Complete usage guide
- [DOCS/MIPS_TRAIN.MD](DOCS/MIPS_TRAIN.MD) - Mip-Splatting training
- [DOCS/SUGAR_USAGE.MD](DOCS/SUGAR_USAGE.MD) - SuGaR mesh extraction

---

## Need Help?

1. Check [DOCS/](DOCS/) folder for detailed guides
2. Review error messages in TensorBoard logs
3. Open an issue on GitHub with:
   - Your GPU model and VRAM
   - Python/PyTorch/CUDA versions
   - Full error traceback
   - Steps to reproduce

---

## License

This project combines multiple codebases. See individual LICENSE files:
- SuGaR: [SuGaR/LICENSE.md](SuGaR/LICENSE.md)
- Mip-Splatting: [mip-splatting/LICENSE.md](mip-splatting/LICENSE.md)
- pytorch3d: [pytorch3d/LICENSE](pytorch3d/LICENSE)
- nvdiffrast: [nvdiffrast/LICENSE.txt](nvdiffrast/LICENSE.txt)
