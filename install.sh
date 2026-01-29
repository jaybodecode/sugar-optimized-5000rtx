#!/bin/bash
# Installation Script - Optimized for RTX 5000 Series
# One-command installer for complete environment setup

set -e  # Exit on error

# Parse command line arguments
AUTO_YES=false
for arg in "$@"; do
    case $arg in
        -y|--yes)
            AUTO_YES=true
            shift
            ;;
        -h|--help)
            echo "Usage: bash install.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -y, --yes    Auto-accept all prompts (non-interactive mode)"
            echo "  -h, --help   Show this help message"
            exit 0
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Status symbols
CHECKMARK="✓"
CROSS="✗"
HOURGLASS="⏳"
ARROW="→"

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Timing helper
STEP_START_TIME=0
start_timer() {
    STEP_START_TIME=$(date +%s)
}

end_timer() {
    local end_time=$(date +%s)
    local duration=$((end_time - STEP_START_TIME))
    if [ $duration -lt 60 ]; then
        echo "${duration}s"
    else
        local mins=$((duration / 60))
        local secs=$((duration % 60))
        echo "${mins}m ${secs}s"
    fi
}

echo "================================================================"
echo "  "
echo "  Optimized for NVIDIA RTX 5000 Series (Blackwell sm_120 ONLY)"
echo "================================================================"
echo ""

# Check system dependencies first
echo -e "${BLUE}${HOURGLASS} [0/9] Checking system dependencies...${NC}"
start_timer

MISSING_DEPS=()

if ! command -v wget &> /dev/null; then
    MISSING_DEPS+=("wget")
fi

if ! command -v git &> /dev/null; then
    MISSING_DEPS+=("git")
fi

if ! command -v gcc &> /dev/null; then
    MISSING_DEPS+=("build-essential")
fi

if ! command -v python3 &> /dev/null; then
    MISSING_DEPS+=("python3")
fi

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo -e "${YELLOW}  ${ARROW} Missing system packages: ${MISSING_DEPS[*]}${NC}"
    echo ""
    if [ "$AUTO_YES" = true ]; then
        response="y"
        echo "  Auto-accepting: Install system packages"
    else
        echo "  Install them now? (requires sudo) (Y/n)"
        read -r response
    fi
    if [[ "$response" =~ ^[Yy]$ ]] || [[ -z "$response" ]]; then
        echo -e "${CYAN}  ${ARROW} Updating package list (apt-get update)...${NC}"
        sudo apt-get update > /dev/null 2>&1
        echo -e "${CYAN}  ${ARROW} Installing packages...${NC}"
        sudo apt-get install -y wget git build-essential python3 > /dev/null 2>&1
        echo -e "${GREEN}  ${CHECKMARK} System dependencies installed${NC}"
    else
        echo -e "${RED}${CROSS} Installation cancelled${NC}"
        echo "  Please install manually: sudo apt-get install wget git build-essential python3"
        exit 1
    fi
else
    echo -e "${GREEN}  ${CHECKMARK} All system dependencies present${NC}"
fi

ELAPSED=$(end_timer)
echo -e "${GREEN}${CHECKMARK} System check complete (took ${ELAPSED})${NC}"
echo ""

# Detect WSL2
IS_WSL2=false
if grep -qi microsoft /proc/version; then
    IS_WSL2=true
    echo -e "${CYAN}${ARROW} Detected: WSL2 environment${NC}"
else
    echo -e "${CYAN}${ARROW} Detected: Native Linux${NC}"
fi

# Check prerequisites
echo ""
echo -e "${BLUE}${HOURGLASS} [1/9] Checking prerequisites...${NC}"
start_timer

# Ensure CUDA is in PATH (if installed but not in PATH)
if ! command -v nvcc &> /dev/null; then
    if [ -d "/usr/local/cuda-13.0/bin" ]; then
        export PATH=/usr/local/cuda-13.0/bin:$PATH
        export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
    elif [ -d "/usr/local/cuda-13.1/bin" ]; then
        export PATH=/usr/local/cuda-13.1/bin:$PATH
        export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:$LD_LIBRARY_PATH
    elif [ -d "/usr/local/cuda/bin" ]; then
        export PATH=/usr/local/cuda/bin:$PATH
        export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    fi
fi

# Check for CUDA (required for both WSL2 and native Linux to compile extensions)
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}  ${CROSS} CUDA toolkit not found (nvcc missing)${NC}"
    echo ""
    echo "  CUDA toolkit 13.0+ is required to compile CUDA extensions."
    echo ""
    if [ "$IS_WSL2" = true ]; then
        echo "  WSL2 Installation:"
        echo "    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb"
        echo "    sudo dpkg -i cuda-keyring_1.1-1_all.deb"
        echo "    sudo apt-get update"
        echo "    sudo apt-get install -y cuda-toolkit-13-0"
        echo "    export PATH=/usr/local/cuda-13.0/bin:\$PATH"
    else
        echo "  Native Linux Installation:"
        echo "    Download from: https://developer.nvidia.com/cuda-downloads"
    fi
    echo ""
    echo "  See INSTALL.md for detailed instructions."
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
echo -e "${GREEN}  ${CHECKMARK} CUDA toolkit version: $CUDA_VERSION${NC}"

# Verify CUDA version
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
if [ "$CUDA_MAJOR" -lt 13 ]; then
    echo -e "${RED}  ${CROSS} CUDA $CUDA_VERSION detected. RTX 5000 series requires CUDA 13.0+${NC}"
    echo "  Please upgrade your CUDA toolkit."
    exit 1
fi

# Check for conda (in PATH or default miniconda3 location)
CONDA_CMD=""
if command -v conda &> /dev/null; then
    CONDA_CMD="conda"
    echo -e "${GREEN}  ${CHECKMARK} Conda found in PATH${NC}"
elif [ -f "$HOME/miniconda3/bin/conda" ]; then
    CONDA_CMD="$HOME/miniconda3/bin/conda"
    # Initialize conda for this script
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    echo -e "${GREEN}  ${CHECKMARK} Conda found at ~/miniconda3${NC}"
else
    # Conda not found, offer to install
    echo -e "${YELLOW}  ${ARROW} Conda not found!${NC}"
    echo ""
    if [ "$AUTO_YES" = true ]; then
        response="y"
        echo "  Auto-accepting: Install Miniconda to ~/miniconda3"
    else
        echo "  Install Miniconda now? This will download ~90MB to ~/miniconda3 (Y/n)"
        read -r response
    fi
    if [[ "$response" =~ ^[Yy]$ ]] || [[ -z "$response" ]]; then
        echo -e "${CYAN}  ${ARROW} Downloading Miniconda...${NC}"
        wget -q --show-progress https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
        echo -e "${CYAN}  ${ARROW} Installing Miniconda to ~/miniconda3...${NC}"
        bash /tmp/miniconda.sh -b -p $HOME/miniconda3
        rm /tmp/miniconda.sh
        
        # Initialize conda for this script
        eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
        
        # Add to bashrc for future sessions
        echo '' >> ~/.bashrc
        echo '# >>> conda initialize >>>' >> ~/.bashrc
        echo "eval \"\$(\$HOME/miniconda3/bin/conda shell.bash hook)\"" >> ~/.bashrc
        echo '# <<< conda initialize <<<' >> ~/.bashrc
        
        # Ask user to accept conda Terms of Service
        echo ""
        echo -e "${YELLOW}  ${ARROW} Conda requires accepting Terms of Service for package channels${NC}"
        echo "     View at: https://www.anaconda.com/terms-of-service"
        echo ""
        if [ "$AUTO_YES" = true ]; then
            tos_response="y"
            echo "  Auto-accepting: Anaconda Terms of Service"
        else
            echo "  Accept Anaconda Terms of Service? (Y/n)"
            read -r tos_response
        fi
        if [[ "$tos_response" =~ ^[Yy]$ ]] || [[ -z "$tos_response" ]]; then
            echo -e "${CYAN}  ${ARROW} Accepting terms for required channels...${NC}"
            $HOME/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
            $HOME/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
            echo -e "${GREEN}  ${CHECKMARK} Terms accepted${NC}"
        else
            echo -e "${RED}${CROSS} Cannot continue without accepting terms${NC}"
            exit 1
        fi
        
        echo -e "${GREEN}  ${CHECKMARK} Miniconda installed!${NC}"
        
        # Set conda command for rest of script
        CONDA_CMD="$HOME/miniconda3/bin/conda"
        eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    else
        echo -e "${RED}${CROSS} Installation cancelled${NC}"
        echo "  Please install conda manually. See INSTALL.md"
        exit 1
    fi
fi

# At this point, conda is available - check TOS
if [ -n "$CONDA_CMD" ]; then
    # Check if TOS already accepted by checking for tos status file
    TOS_FILE="$HOME/.conda/.tos_status"
    if [ ! -f "$TOS_FILE" ]; then
        echo ""
        echo -e "${YELLOW}  ${ARROW} Conda requires accepting Terms of Service${NC}"
        echo "     View at: https://www.anaconda.com/terms-of-service"
        echo ""
        if [ "$AUTO_YES" = true ]; then
            tos_response="y"
            echo "  Auto-accepting: Anaconda Terms of Service"
        else
            echo "  Accept Anaconda Terms of Service? (Y/n)"
            read -r tos_response
        fi
        if [[ "$tos_response" =~ ^[Yy]$ ]] || [[ -z "$tos_response" ]]; then
            echo -e "${CYAN}  ${ARROW} Accepting terms for required channels...${NC}"
            $CONDA_CMD tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
            $CONDA_CMD tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
            # Mark as accepted
            mkdir -p "$HOME/.conda"
            touch "$TOS_FILE"
            echo -e "${GREEN}  ${CHECKMARK} Terms accepted${NC}"
        else
            echo -e "${RED}${CROSS} Cannot continue without accepting terms${NC}"
            exit 1
        fi
    fi
fi

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}  ${CROSS} nvidia-smi not found!${NC}"
    if [ "$IS_WSL2" = true ]; then
        echo "  WSL2 requires NVIDIA drivers 570+ installed on Windows."
        echo "  Download: https://www.nvidia.com/Download/index.aspx"
    else
        echo "  Please install NVIDIA drivers 570+."
    fi
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo -e "${GREEN}  ${CHECKMARK} GPU detected: $GPU_NAME${NC}"

# Enforce RTX 5000 series only
if [[ ! "$GPU_NAME" =~ "RTX 50" ]]; then
    echo ""
    echo -e "${RED}  ${CROSS} This project ONLY supports RTX 5000 Series (Blackwell architecture)${NC}"
    echo "     Your GPU: $GPU_NAME"
    echo ""
    echo "     RTX 5000 series uses sm_120 (compute capability 12.0) and Blackwell-specific"
    echo "     optimizations that are not compatible with older GPU architectures."
    echo ""
    echo "     Supported GPUs: RTX 5060, 5060 Ti, 5070, 5070 Ti, 5080, 5090"
    echo ""
    exit 1
fi

echo -e "${GREEN}  ${CHECKMARK} RTX 5000 Series verified - compatible architecture${NC}"

ELAPSED=$(end_timer)
echo -e "${GREEN}${CHECKMARK} Prerequisites check complete (took ${ELAPSED})${NC}"

# Create conda environment
echo ""
echo -e "${BLUE}${HOURGLASS} [2/9] Creating conda environment 'sugar'...${NC}"
start_timer

if conda env list | grep -q "^sugar "; then
    echo -e "${YELLOW}  ${ARROW} Environment 'sugar' already exists${NC}"
    if [ "$AUTO_YES" = true ]; then
        response="n"
        echo "  Auto-skipping: Using existing environment"
    else
        echo "  Remove and recreate? (y/N)"
        read -r response
    fi
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo -e "${CYAN}  ${ARROW} Removing existing environment...${NC}"
        conda env remove -n sugar -y > /dev/null 2>&1
        conda create -n sugar python=3.12 -y -q
        echo -e "${GREEN}  ${CHECKMARK} Environment recreated${NC}"
    else
        echo -e "${GREEN}  ${CHECKMARK} Using existing environment${NC}"
    fi
else
    echo -e "${CYAN}  ${ARROW} Creating environment with Python 3.12...${NC}"
    conda create -n sugar python=3.12 -y -q
    echo -e "${GREEN}  ${CHECKMARK} Environment created${NC}"
fi

ELAPSED=$(end_timer)
echo -e "${GREEN}${CHECKMARK} Conda environment ready (took ${ELAPSED})${NC}"

# Activate environment
echo ""
echo -e "${BLUE}${HOURGLASS} [3/9] Activating environment...${NC}"
start_timer

# Source conda in this script (ensure it's properly initialized)
if [ "$CONDA_CMD" = "conda" ]; then
    eval "$(conda shell.bash hook)"
else
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate sugar

echo -e "${GREEN}  ${CHECKMARK} Environment 'sugar' activated${NC}"
ELAPSED=$(end_timer)
echo -e "${GREEN}${CHECKMARK} Activation complete (took ${ELAPSED})${NC}"

# Install PyTorch
echo ""
echo -e "${BLUE}${HOURGLASS} [4/9] Installing PyTorch with CUDA 13.0 (nightly)...${NC}"
start_timer

if pip list 2>/dev/null | grep -q "^torch "; then
    echo -e "${GREEN}  ${CHECKMARK} PyTorch already installed - skipping${NC}"
else
    echo -e "${CYAN}  ${ARROW} This may take few minutes depending on your connection${NC}"
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130 -q
    echo -e "${GREEN}  ${CHECKMARK} PyTorch installed${NC}"
fi

ELAPSED=$(end_timer)
echo -e "${GREEN}${CHECKMARK} PyTorch installation complete (took ${ELAPSED})${NC}"

# Install conda dependencies
echo ""
echo -e "${BLUE}${HOURGLASS} [5/9] Installing conda dependencies (fvcore, iopath, ninja)...${NC}"
start_timer

if conda list 2>/dev/null | grep -q "fvcore" && conda list 2>/dev/null | grep -q "iopath" && conda list 2>/dev/null | grep -q "ninja"; then
    echo -e "${GREEN}  ${CHECKMARK} fvcore, iopath, and ninja already installed - skipping${NC}"
else
    conda install -c fvcore -c iopath -c conda-forge fvcore iopath ninja -y -q
    echo -e "${GREEN}  ${CHECKMARK} fvcore, iopath, and ninja installed${NC}"
fi

ELAPSED=$(end_timer)
echo -e "${GREEN}${CHECKMARK} Conda dependencies complete (took ${ELAPSED})${NC}"

# Install Python packages
echo ""
echo -e "${BLUE}${HOURGLASS} [6/9] Installing Python packages from requirements.txt...${NC}"
start_timer

cd "$SCRIPT_DIR"
pip install -r requirements.txt -q

echo -e "${GREEN}  ${CHECKMARK} All Python packages installed${NC}"
ELAPSED=$(end_timer)
echo -e "${GREEN}${CHECKMARK} Python packages complete (took ${ELAPSED})${NC}"

# Compile CUDA extensions
echo ""
echo -e "${BLUE}${HOURGLASS} [7/9] Compiling CUDA extensions...${NC}"
echo -e "${CYAN}  ${ARROW} This will take several minutes or more (pytorch3d, nvdiffrast, simple-knn, diff-gaussian-rasterization)${NC}"
start_timer

# Auto-detect compute capability (RTX 5000 series = sm_120)
echo -e "${CYAN}  ${ARROW} Detecting GPU compute capability...${NC}"
COMPUTE_CAP=$(python -c "import torch; cc = torch.cuda.get_device_capability(0); print(f'{cc[0]}.{cc[1]}')" 2>/dev/null || echo "12.0")
echo -e "${GREEN}  ${CHECKMARK} Compute capability: sm_${COMPUTE_CAP/./}${NC}"

# Set CUDA_HOME for compilation
if [ -z "$CUDA_HOME" ]; then
    CUDA_HOME=$(dirname $(dirname $(which nvcc)))
fi
echo -e "${GREEN}  ${CHECKMARK} CUDA_HOME: $CUDA_HOME${NC}"

export CUDA_HOME
export TORCH_CUDA_ARCH_LIST="$COMPUTE_CAP"
export FORCE_CUDA=1
export MAX_JOBS=8

# Compile pytorch3d
echo ""
echo -e "${CYAN}  ${ARROW} [7a/7] Compiling pytorch3d (mesh operations & KNN)...${NC}"

# Check if already installed
if pip list 2>/dev/null | grep -q "pytorch3d"; then
    echo -e "${GREEN}  ${CHECKMARK} pytorch3d already installed - skipping${NC}"
else
    cd "$SCRIPT_DIR/pytorch3d"
    if pip install --no-build-isolation -e . > /tmp/pytorch3d_install.log 2>&1; then
        echo -e "${GREEN}  ${CHECKMARK} pytorch3d compiled${NC}"
    else
        echo -e "${RED}  ${CROSS} pytorch3d compilation failed${NC}"
        echo "  See /tmp/pytorch3d_install.log for details"
        tail -20 /tmp/pytorch3d_install.log
        exit 1
    fi
fi

# Compile nvdiffrast
echo -e "${CYAN}  ${ARROW} [7b/7] Compiling nvdiffrast (GPU rasterization)...${NC}"
if pip list 2>/dev/null | grep -q "nvdiffrast"; then
    echo -e "${GREEN}  ${CHECKMARK} nvdiffrast already installed - skipping${NC}"
else
    cd "$SCRIPT_DIR/nvdiffrast"
    if pip install --no-build-isolation . > /tmp/nvdiffrast_install.log 2>&1; then
        echo -e "${GREEN}  ${CHECKMARK} nvdiffrast compiled${NC}"
    else
        echo -e "${RED}  ${CROSS} nvdiffrast compilation failed${NC}"
        echo "  See /tmp/nvdiffrast_install.log for details"
        tail -20 /tmp/nvdiffrast_install.log
        exit 1
    fi
fi

# Compile simple-knn (for mip-splatting)
echo -e "${CYAN}  ${ARROW} [7c/7] Compiling simple-knn (Gaussian Splatting)...${NC}"
if pip list 2>/dev/null | grep -q "simple_knn"; then
    echo -e "${GREEN}  ${CHECKMARK} simple-knn already installed - skipping${NC}"
else
    cd "$SCRIPT_DIR/mip-splatting/submodules/simple-knn"
    if python install_rtx50.py install > /tmp/simple_knn_install.log 2>&1; then
        echo -e "${GREEN}  ${CHECKMARK} simple-knn compiled${NC}"
    else
        echo -e "${RED}  ${CROSS} simple-knn compilation failed${NC}"
        echo "  See /tmp/simple_knn_install.log for details"
        tail -20 /tmp/simple_knn_install.log
        exit 1
    fi
fi

# Compile diff-gaussian-rasterization (for mip-splatting)
echo -e "${CYAN}  ${ARROW} [7d/7] Compiling diff-gaussian-rasterization...${NC}"
if pip list 2>/dev/null | grep -q "diff_gaussian_rasterization"; then
    echo -e "${GREEN}  ${CHECKMARK} diff-gaussian-rasterization already installed - skipping${NC}"
else
    cd "$SCRIPT_DIR/mip-splatting/submodules/diff-gaussian-rasterization"
    if python install_rtx50.py install > /tmp/diff_gauss_install.log 2>&1; then
        echo -e "${GREEN}  ${CHECKMARK} diff-gaussian-rasterization compiled${NC}"
    else
        echo -e "${RED}  ${CROSS} diff-gaussian-rasterization compilation failed${NC}"
        echo "  See /tmp/diff_gauss_install.log for details"
        tail -20 /tmp/diff_gauss_install.log
        exit 1
    fi
fi

ELAPSED=$(end_timer)
echo -e "${GREEN}${CHECKMARK} CUDA extensions compiled (took ${ELAPSED})${NC}"

# Run verification
echo ""
echo -e "${BLUE}${HOURGLASS} [8/9] Running verification tests...${NC}"
start_timer

cd "$SCRIPT_DIR"
python verify_install.py

VERIFY_EXIT=$?
ELAPSED=$(end_timer)

if [ $VERIFY_EXIT -eq 0 ]; then
    echo -e "${GREEN}${CHECKMARK} Verification tests passed (took ${ELAPSED})${NC}"
else
    echo -e "${YELLOW}${ARROW} Verification completed with warnings (took ${ELAPSED})${NC}"
fi

echo ""
echo -e "${BLUE}${HOURGLASS} [9/9] Installation summary...${NC}"
echo ""
echo "  ┌─────────────────────────────────────────────────────┐"
if [ "$IS_WSL2" = true ]; then
    echo "  │ Environment: WSL2                                   │"
else
    echo "  │ Environment: Native Linux                           │"
fi
echo "  │ GPU: ${GPU_NAME:0:43}$(printf '%*s' $((43 - ${#GPU_NAME})) '') │"
echo "  │ Compute: sm_${COMPUTE_CAP/./}                                      │"
PYTHON_VER=$(python --version | cut -d' ' -f2)
echo "  │ Python: ${PYTHON_VER}                                    │"
PYTORCH_VER=$(python -c 'import torch; print(torch.__version__)' 2>/dev/null | cut -d'+' -f1)
echo "  │ PyTorch: ${PYTORCH_VER:0:41}$(printf '%*s' $((41 - ${#PYTORCH_VER})) '') │"
echo "  └─────────────────────────────────────────────────────┘"
echo ""

if [ $VERIFY_EXIT -eq 0 ]; then
    echo "================================================================"
    echo -e "${GREEN}${CHECKMARK} Installation complete!${NC}"
    echo "================================================================"
    echo ""
    echo "To activate the environment:"
    echo "  ${CYAN}conda activate sugar${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Prepare your dataset (COLMAP or NeRF format)"
    echo "  2. See ${CYAN}DOCS/MIPS_TRAIN.MD${NC} for Mip-Splatting training"
    echo "  3. See ${CYAN}DOCS/SUGAR_USAGE.MD${NC} for SuGaR mesh extraction"
    echo "  4. See ${CYAN}README.md${NC} for project overview"
    echo ""
else
    echo "================================================================"
    echo -e "${YELLOW}${ARROW} Installation completed with warnings${NC}"
    echo "================================================================"
    echo ""
    echo "Some tests failed. Please review the output above."
    echo "You may still be able to use the software, but some features"
    echo "may not work correctly."
    echo ""
    echo "For troubleshooting, see:"
    echo "  ${CYAN}INSTALL.md${NC} - Troubleshooting section"
    echo "  ${CYAN}README.md${NC} - Known issues"
    echo ""
fi
