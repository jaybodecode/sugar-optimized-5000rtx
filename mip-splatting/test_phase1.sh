#!/bin/bash
# Phase 1 Testing Script - Register Spilling Optimization

set -e  # Exit on error

echo "======================================"
echo "Phase 1: Register Spilling (500MB)"
echo "======================================"
echo ""

# Step 1: Recompile CUDA extension
echo "Step 1: Recompiling diff-gaussian-rasterization with __launch_bounds__..."
cd /home/jason/GITHUB/SugarV3/mip-splatting/submodules/diff-gaussian-rasterization/
pip install -e . --no-build-isolation

if [ $? -eq 0 ]; then
    echo "✓ Compilation successful!"
else
    echo "❌ Compilation failed!"
    exit 1
fi

echo ""
echo "Step 2: Finding latest checkpoint..."
cd /home/jason/GITHUB/SugarV3/mip-splatting/

# Find most recent checkpoint
CHECKPOINT=$(find ../SAMPLES/garden_output/ -name "chkpnt*.pth" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2)

if [ -z "$CHECKPOINT" ]; then
    echo "❌ No checkpoint found in ../SAMPLES/garden_output/"
    echo "   Please ensure you have a checkpoint from a previous training run"
    exit 1
fi

echo "✓ Found checkpoint: $CHECKPOINT"
echo ""

# Step 3: Run baseline test
echo "Step 3: Measuring VRAM with Phase 1 optimization..."
python test_vram_baseline.py \
    --checkpoint "$CHECKPOINT" \
    --dataset ../SAMPLES/garden \
    --label "Phase 1: Register Spilling"

echo ""
echo "======================================"
echo "Next Steps:"
echo "======================================"
echo "1. Record the Peak VRAM value above"
echo "2. Compare with your baseline measurement"
echo "3. Expected savings: ~200-500 MB"
echo ""
echo "To commit this change:"
echo "  cd /home/jason/GITHUB/SugarV3"
echo "  git add -A"
echo "  git commit -m 'Phase 1: Register spilling optimization (500MB VRAM)'"
echo ""
echo "To proceed to Phase 2:"
echo "  bash test_phase2.sh"
