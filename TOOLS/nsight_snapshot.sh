#!/bin/bash
# Capture GPU memory snapshot with NVIDIA Nsight Systems
# This version profiles system-wide GPU activity for a short duration

echo "🔍 Checking for active training process..."
PID=$(ps aux | grep "[t]rain.py" | awk '{print $2}' | head -1)

if [ -z "$PID" ]; then
    echo "⚠️  No training process found (looking for train.py)"
    echo "    This will still capture all GPU activity system-wide"
else
    echo "✅ Found training at PID: $PID"
    PROC_INFO=$(ps -p $PID -o command= | cut -c1-80)
    echo "   Process: $PROC_INFO"
fi
echo ""

# Create output directory
OUTPUT_DIR="/home/jason/GITHUB/SugarV3/SuGaR/TOOLS/profiles"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="$OUTPUT_DIR/training_profile_${TIMESTAMP}.nsys-rep"

DURATION="${1:-30}"  # Default 30 seconds, or pass duration as argument

echo "📊 Profiling GPU activity for ${DURATION} seconds..."
echo "   Mode: system-wide (captures all GPU processes)"
echo "   Output: $OUTPUT_FILE"
echo ""

# System-wide profiling (captures all GPU activity)
nsys profile \
  --output="$OUTPUT_FILE" \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --sample=cpu \
  --cpuctxsw=none \
  --duration="$DURATION" \
  --delay=0 \
  --capture-range=none \
  --stats=true \
  echo "Profiling GPU for ${DURATION}s..."

echo ""
echo "✅ Profile captured!"
echo ""

# Check if file was created
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "❌ Profile file not created. Check permissions or disk space."
    exit 1
fi

FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
echo "📦 File size: $FILE_SIZE"
echo ""

echo "📂 View with:"
echo "   nsys-ui $OUTPUT_FILE"
echo ""

echo "📊 Quick stats (Memory):"
nsys stats --report cuda_gpu_mem_size_sum "$OUTPUT_FILE" 2>/dev/null | head -20 || echo "  (stats not available)"
echo ""

echo "📊 Quick stats (Kernels):"
nsys stats --report cuda_gpu_kern_sum "$OUTPUT_FILE" 2>/dev/null | head -15 || echo "  (stats not available)"
echo ""

echo "💾 Export to SQLite for analysis:"
echo "   nsys export --type=sqlite --output=${OUTPUT_FILE%.nsys-rep}.sqlite $OUTPUT_FILE"
echo ""

echo "✅ Done! Profile saved at: $OUTPUT_FILE"
