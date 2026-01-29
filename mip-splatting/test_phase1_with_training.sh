#!/bin/bash
# Phase 1 Test - With Real Training (Not Just Eval)
# Tests register spilling optimization during actual training scenario

set -e

echo "======================================"
echo "Phase 1: Register Spilling Test"
echo "======================================"
echo ""

cd /home/jason/GITHUB/SugarV3/mip-splatting/

# Find baseline checkpoint
CHECKPOINT=$(find ../SAMPLES/garden_output/baseline-r4-test* -name "chkpnt3000.pth" -type f | head -1)

if [ -z "$CHECKPOINT" ]; then
    echo "❌ Baseline checkpoint not found!"
    echo "   Run: bash create_baseline_checkpoint.sh first"
    exit 1
fi

echo "✓ Found checkpoint: $CHECKPOINT"
echo ""

# Step 1: Recompile with optimization
echo "Step 1: Recompiling diff-gaussian-rasterization..."
cd /home/jason/GITHUB/SugarV3/mip-splatting/submodules/diff-gaussian-rasterization/
pip install -e . --no-build-isolation

if [ $? -ne 0 ]; then
    echo "❌ Compilation failed!"
    exit 1
fi

echo "✓ Compilation successful!"
echo ""

cd /home/jason/GITHUB/SugarV3/mip-splatting/

# Step 2: Continue training for 200 more iterations (3000 → 3200)
# This will trigger test_iteration at 3100
echo "Step 2: Running training 3000→3200 (includes test at 3100)..."
echo "Watch VRAM in nvidia-smi during this run!"
echo ""
echo "IMPORTANT: Open another terminal and run:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Starting in 3 seconds..."
sleep 3

python train.py \
  -s ../SAMPLES/garden \
  -r 4 \
  --iteration 3200 \
  --start_checkpoint "$CHECKPOINT" \
  --checkpoint_iterations 3200 \
  --test_iterations 3100 \
  --save_iterations 3200 \
  --experiment_name "phase1-test" \
  2>&1 | tee phase1-training.log

echo ""
echo "======================================"
echo "Phase 1 Results:"
echo "======================================"
echo ""
echo "Review the console output above for:"
echo "  - Peak VRAM during training (before iter 3100)"
echo "  - Peak VRAM during test (at iter 3100)"
echo "  - Final VRAM after test"
echo ""
echo "Compare with baseline run to see VRAM savings!"
echo ""
echo "Expected savings: 200-500 MB reduction in peak VRAM"
echo ""
echo "To commit if successful:"
echo "  cd /home/jason/GITHUB/SugarV3"
echo "  git add -A"
echo "  git commit -m 'Phase 1: Register spilling (500MB VRAM)'"
echo ""
