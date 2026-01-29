#!/bin/bash
# Profile training from start with NVIDIA Nsight Systems
# Usage: ./nsight_profile_training.sh <your normal training command>
# Example: ./nsight_profile_training.sh python train.py -s ../SAMPLES/garden ...

OUTPUT_DIR="/home/jason/GITHUB/SugarV3/SuGaR/TOOLS/profiles"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="$OUTPUT_DIR/training_full_${TIMESTAMP}.nsys-rep"

echo "🚀 Starting training with profiling..."
echo "   Output: $OUTPUT_FILE"
echo "   Command: $@"
echo ""

nsys profile \
  --output="$OUTPUT_FILE" \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --sample=cpu \
  --cpuctxsw=process-tree \
  --stats=true \
  "$@"

echo ""
echo "✅ Training complete! Profile saved."
echo ""
echo "📂 View with:"
echo "   nsys-ui $OUTPUT_FILE"
echo ""
echo "📊 Generate stats:"
echo "   nsys stats $OUTPUT_FILE"
