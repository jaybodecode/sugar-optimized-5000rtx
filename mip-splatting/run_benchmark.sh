#!/bin/bash
# Run Benchmark Test - Trains for 1000 iterations and measures performance
# Usage: bash run_benchmark.sh <label>
# Example: bash run_benchmark.sh baseline
#          bash run_benchmark.sh phase1

set -e

if [ -z "$1" ]; then
    echo "Usage: bash run_benchmark.sh <label>"
    echo "Example: bash run_benchmark.sh baseline"
    exit 1
fi

LABEL=$1
LOGFILE="benchmark_logs/${LABEL}.log"
REPORTFILE="benchmark_reports/${LABEL}.txt"

echo "======================================"
echo "Running Benchmark: $LABEL"
echo "======================================"
echo ""

cd /home/jason/GITHUB/SugarV3/mip-splatting/

# Find baseline checkpoint
CHECKPOINT=$(find ../SAMPLES/garden_output/baseline-r4-3k* -name "chkpnt3000.pth" -type f | head -1)

if [ -z "$CHECKPOINT" ]; then
    echo "❌ Baseline checkpoint not found!"
    echo "   Run: bash create_baseline_checkpoint.sh first"
    exit 1
fi

echo "✓ Using checkpoint: $CHECKPOINT"
echo "✓ Log file: $LOGFILE"
echo "✓ Report file: $REPORTFILE"
echo ""
echo "Training 3000 → 4000 iterations (includes test at 3100)"
echo "This will take ~4-5 minutes"
echo ""
echo "IMPORTANT: Open another terminal and run:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Starting in 3 seconds..."
sleep 3

# Record start time
START_TIME=$(date +%s)

# Run training and capture output
conda run -n rtx5000_fresh python train.py \
  -s ../SAMPLES/garden \
  -r 4 \
  --eval \
  --iteration 4000 \
  --start_checkpoint "$CHECKPOINT" \
  --checkpoint_iterations 4000 \
  --test_iterations 3100 \
  --save_iterations 4000 \
  --experiment_name "benchmark-${LABEL}" \
  2>&1 | tee "$LOGFILE"

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "✓ Benchmark complete!"
echo ""

# Generate report
conda run -n rtx5000_fresh bash analyze_benchmark.sh "$LABEL"

echo ""
echo "======================================"
echo "Quick Summary:"
echo "======================================"
cat "$REPORTFILE"
echo ""
