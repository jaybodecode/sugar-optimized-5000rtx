#!/bin/bash
# Create Baseline Checkpoint for VRAM Testing (RUN ONCE)
# Uses -r4 (quarter resolution) to ensure it fits in VRAM

set -e

echo "======================================"
echo "Creating Baseline Checkpoint (-r4)"
echo "======================================"
echo ""
echo "This creates a checkpoint for benchmarking (RUN ONCE)"
echo "Resolution: -r4 (quarter res)"
echo "Iterations: 3000"
echo "Expected time: ~8-10 minutes"
echo "Expected peak VRAM: ~8-10GB"
echo ""

cd /home/jason/GITHUB/SugarV3/mip-splatting/

# Create directories for benchmark results
mkdir -p benchmark_logs
mkdir -p benchmark_reports

# Clean any previous baseline checkpoint
rm -rf ../SAMPLES/garden_output/baseline-r4-3k

echo "Starting baseline training..."
conda run -n rtx5000_fresh python train.py \
  -s ../SAMPLES/garden \
  -r 4 \
  --iteration 3000 \
  --checkpoint_iterations 3000 \
  --test_iterations 3100 \
  --save_iterations 3000 \
  --experiment_name "baseline-r4-3k" \
  2>&1 | tee benchmark_logs/baseline-creation.log

echo ""

# Find the checkpoint
CHECKPOINT=$(find ../SAMPLES/garden_output/baseline-r4-3k* -name "chkpnt3000.pth" -type f | head -1)

if [ -z "$CHECKPOINT" ]; then
    echo "❌ Checkpoint not found!"
    exit 1
fi

echo "✓ Baseline checkpoint created: $CHECKPOINT"
echo ""
echo "======================================"
echo "Next Steps:"
echo "======================================"
echo "1. Run baseline benchmark:"
echo "   bash run_benchmark.sh baseline"
echo ""
echo "2. Make code changes (e.g., Phase 1 optimization)"
echo ""
echo "3. Run optimized benchmark:"
echo "   bash run_benchmark.sh phase1"
echo ""
echo "4. Compare results:"
echo "   bash compare_benchmarks.sh baseline phase1"
echo ""
