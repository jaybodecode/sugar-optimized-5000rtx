#!/bin/bash
# Analyze Benchmark Log - Extract VRAM and timing data
# Usage: bash analyze_benchmark.sh <label>

if [ -z "$1" ]; then
    echo "Usage: bash analyze_benchmark.sh <label>"
    exit 1
fi

LABEL=$1
LOGFILE="benchmark_logs/${LABEL}.log"
REPORTFILE="benchmark_reports/${LABEL}.txt"

if [ ! -f "$LOGFILE" ]; then
    echo "❌ Log file not found: $LOGFILE"
    exit 1
fi

echo "Analyzing benchmark: $LABEL"
echo ""

# Strip ANSI color codes for parsing
CLEAN_LOG=$(sed 's/\x1b\[[0-9;]*m//g' "$LOGFILE")

# Extract key metrics from log
# Progress bar format: ⠇ Training ━╸━━━━━━━━  17% 5000/30001 L:0.0762 3.6it/s V:62%/9.9GB R:57%/1

# Get VRAM values (format: V:62%/9.9GB)
VRAM_VALUES=$(echo "$CLEAN_LOG" | grep -oP 'V:\d+%/\K[0-9.]+(?=GB)' | sort -n || true)

if [ -z "$VRAM_VALUES" ]; then
    echo "⚠ Warning: No VRAM data found in log (expected format: V:XX%/X.XGB)"
    MIN_VRAM="N/A"
    MAX_VRAM="N/A"
    AVG_VRAM="N/A"
else
    MIN_VRAM=$(echo "$VRAM_VALUES" | head -1)
    MAX_VRAM=$(echo "$VRAM_VALUES" | sort -n | tail -1)
    AVG_VRAM=$(echo "$VRAM_VALUES" | awk '{sum+=$1; count++} END {printf "%.1f", sum/count}')
fi

# Extract VRAM percentages for reference
VRAM_PERCENT_VALUES=$(echo "$CLEAN_LOG" | grep -oP 'V:\K\d+(?=%/)' | sort -n || true)
if [ -n "$VRAM_PERCENT_VALUES" ]; then
    MAX_VRAM_PCT=$(echo "$VRAM_PERCENT_VALUES" | tail -1)
else
    MAX_VRAM_PCT="N/A"
fi

# Extract training speed (format: 3.6it/s)
SPEED_VALUES=$(echo "$CLEAN_LOG" | grep -oP '[0-9.]+(?=it/s)' || true)
if [ -z "$SPEED_VALUES" ]; then
    AVG_SPEED="N/A"
else
    AVG_SPEED=$(echo "$SPEED_VALUES" | awk '{sum+=$1; count++} END {printf "%.2f", sum/count}')
fi

# Extract loss values (format: L:0.0762)
LOSS_VALUES=$(echo "$CLEAN_LOG" | grep -oP 'L:\K[0-9.]+' || true)
if [ -n "$LOSS_VALUES" ]; then
    FINAL_LOSS=$(echo "$LOSS_VALUES" | tail -1)
else
    FINAL_LOSS="N/A"
fi

# Count iterations completed (format: 5000/30001)
ITERATIONS=$(echo "$CLEAN_LOG" | grep -oP '\d+(?=/\d+\s+L:)' | tail -1 || echo "N/A")

# Extract test iteration VRAM peak (look for VRAM spike during "Evaluating" messages)
# During test, VRAM typically shows higher values
TEST_SECTION=$(echo "$CLEAN_LOG" | grep -A 20 "Evaluating test set" || echo "")
if [ -n "$TEST_SECTION" ]; then
    TEST_VRAM=$(echo "$TEST_SECTION" | grep -oP 'V:\d+%/\K[0-9.]+(?=GB)' | sort -n | tail -1 || echo "N/A")
else
    TEST_VRAM="N/A"
fi

# Calculate elapsed time from log timestamps
FIRST_TIMESTAMP=$(grep -oP '\d{2}:\d{2}:\d{2}' "$LOGFILE" | head -1)
LAST_TIMESTAMP=$(grep -oP '\d{2}:\d{2}:\d{2}' "$LOGFILE" | tail -1)

# Extract quality metrics from Evaluation Results table
# Look for pattern like: │  TRAIN  │ 0.035384 │     26.01 │ 0.7890 │ 0.2412 │
EVAL_LINE=$(echo "$CLEAN_LOG" | grep -E '│\s+(TRAIN|TEST)\s+│' | tail -1 || echo "")
if [ -n "$EVAL_LINE" ]; then
    # Extract values (format: │ DATASET │ L1 │ PSNR │ SSIM │ LPIPS │)
    EVAL_DATASET=$(echo "$EVAL_LINE" | awk -F'│' '{print $2}' | xargs)
    EVAL_L1=$(echo "$EVAL_LINE" | awk -F'│' '{print $3}' | xargs)
    EVAL_PSNR=$(echo "$EVAL_LINE" | awk -F'│' '{print $4}' | xargs)
    EVAL_SSIM=$(echo "$EVAL_LINE" | awk -F'│' '{print $5}' | xargs)
    EVAL_LPIPS=$(echo "$EVAL_LINE" | awk -F'│' '{print $6}' | xargs)
else
    EVAL_DATASET="N/A"
    EVAL_L1="N/A"
    EVAL_PSNR="N/A"
    EVAL_SSIM="N/A"
    EVAL_LPIPS="N/A"
fi

# Generate report
cat > "$REPORTFILE" <<EOF
====================================
Benchmark Report: $LABEL
====================================
Date: $(date)
Log: $LOGFILE

VRAM Usage (Training):
  Minimum: ${MIN_VRAM} GB
  Maximum: ${MAX_VRAM} GB (${MAX_VRAM_PCT}%)
  Average: ${AVG_VRAM} GB

VRAM Usage (Test):
  Peak during test_iteration: ${TEST_VRAM} GB

Performance:
  Average Speed: ${AVG_SPEED} it/s
  Final Loss: ${FINAL_LOSS}
  Iterations Completed: ${ITERATIONS}
  Duration: ${FIRST_TIMESTAMP} - ${LAST_TIMESTAMP}

Quality Metrics (Evaluation):
  Dataset: ${EVAL_DATASET}
  L1 Loss: ${EVAL_L1}
  PSNR: ${EVAL_PSNR} dB
  SSIM: ${EVAL_SSIM}
  LPIPS: ${EVAL_LPIPS}

Log File: $LOGFILE
====================================
EOF

echo "✓ Report saved to: $REPORTFILE"
