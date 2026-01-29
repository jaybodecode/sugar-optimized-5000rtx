#!/bin/bash
# Compare Two Benchmark Reports
# Usage: bash compare_benchmarks.sh <baseline_label> <optimized_label>

set -e

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: bash compare_benchmarks.sh <baseline_label> <optimized_label>"
    echo "Example: bash compare_benchmarks.sh baseline phase1"
    exit 1
fi

BASELINE_LABEL=$1
OPTIMIZED_LABEL=$2

BASELINE_REPORT="benchmark_reports/${BASELINE_LABEL}.txt"
OPTIMIZED_REPORT="benchmark_reports/${OPTIMIZED_LABEL}.txt"

if [ ! -f "$BASELINE_REPORT" ]; then
    echo "❌ Baseline report not found: $BASELINE_REPORT"
    exit 1
fi

if [ ! -f "$OPTIMIZED_REPORT" ]; then
    echo "❌ Optimized report not found: $OPTIMIZED_REPORT"
    exit 1
fi

echo "======================================"
echo "Benchmark Comparison"
echo "======================================"
echo ""

# Extract metrics from both reports
extract_metric() {
    local file=$1
    local pattern=$2
    grep "$pattern" "$file" | grep -oP '[0-9.]+' | head -1
}

# Baseline metrics
BASELINE_MAX=$(extract_metric "$BASELINE_REPORT" "Maximum:")
BASELINE_AVG=$(extract_metric "$BASELINE_REPORT" "Average:")
BASELINE_TEST=$(extract_metric "$BASELINE_REPORT" "Peak during test")
BASELINE_SPEED=$(extract_metric "$BASELINE_REPORT" "Average Speed:")

# Optimized metrics
OPTIMIZED_MAX=$(extract_metric "$OPTIMIZED_REPORT" "Maximum:")
OPTIMIZED_AVG=$(extract_metric "$OPTIMIZED_REPORT" "Average:")
OPTIMIZED_TEST=$(extract_metric "$OPTIMIZED_REPORT" "Peak during test")
OPTIMIZED_SPEED=$(extract_metric "$OPTIMIZED_REPORT" "Average Speed:")

# Calculate differences (if values exist)
if [ -n "$BASELINE_MAX" ] && [ -n "$OPTIMIZED_MAX" ]; then
    MAX_DIFF=$(echo "$BASELINE_MAX - $OPTIMIZED_MAX" | bc)
    MAX_DIFF_MB=$(echo "$MAX_DIFF * 1024" | bc | xargs printf "%.0f")
else
    MAX_DIFF="N/A"
    MAX_DIFF_MB="N/A"
fi

if [ -n "$BASELINE_AVG" ] && [ -n "$OPTIMIZED_AVG" ]; then
    AVG_DIFF=$(echo "$BASELINE_AVG - $OPTIMIZED_AVG" | bc)
    AVG_DIFF_MB=$(echo "$AVG_DIFF * 1024" | bc | xargs printf "%.0f")
else
    AVG_DIFF="N/A"
    AVG_DIFF_MB="N/A"
fi

if [ -n "$BASELINE_TEST" ] && [ -n "$OPTIMIZED_TEST" ]; then
    TEST_DIFF=$(echo "$BASELINE_TEST - $OPTIMIZED_TEST" | bc)
    TEST_DIFF_MB=$(echo "$TEST_DIFF * 1024" | bc | xargs printf "%.0f")
else
    TEST_DIFF="N/A"
    TEST_DIFF_MB="N/A"
fi

if [ -n "$BASELINE_SPEED" ] && [ -n "$OPTIMIZED_SPEED" ]; then
    SPEED_DIFF=$(echo "$OPTIMIZED_SPEED - $BASELINE_SPEED" | bc)
    SPEED_PCT=$(echo "scale=1; ($SPEED_DIFF / $BASELINE_SPEED) * 100" | bc)
else
    SPEED_DIFF="N/A"
    SPEED_PCT="N/A"
fi

# Print comparison
cat <<EOF
Baseline:  $BASELINE_LABEL
Optimized: $OPTIMIZED_LABEL

VRAM Usage (Training):
                    Baseline    Optimized   Difference
  Max VRAM:         ${BASELINE_MAX:-N/A} GB       ${OPTIMIZED_MAX:-N/A} GB      ${MAX_DIFF:-N/A} GB (${MAX_DIFF_MB:-N/A} MB)
  Avg VRAM:         ${BASELINE_AVG:-N/A} GB       ${OPTIMIZED_AVG:-N/A} GB      ${AVG_DIFF:-N/A} GB (${AVG_DIFF_MB:-N/A} MB)

VRAM Usage (Test):
  Test Peak:        ${BASELINE_TEST:-N/A} GB       ${OPTIMIZED_TEST:-N/A} GB      ${TEST_DIFF:-N/A} GB (${TEST_DIFF_MB:-N/A} MB)

Performance:
  Speed:            ${BASELINE_SPEED:-N/A} it/s    ${OPTIMIZED_SPEED:-N/A} it/s   ${SPEED_DIFF:-N/A} it/s (${SPEED_PCT:-N/A}%)

====================================
Summary:
====================================
EOF

# Interpret results
if [ "$MAX_DIFF_MB" != "N/A" ]; then
    if [ "$MAX_DIFF_MB" -gt 0 ]; then
        echo "✓ VRAM Reduction: ${MAX_DIFF_MB} MB saved"
    else
        echo "⚠ VRAM Increase: $(echo "$MAX_DIFF_MB * -1" | bc) MB more used"
    fi
fi

if [ "$SPEED_PCT" != "N/A" ]; then
    SPEED_PCT_NUM=$(echo "$SPEED_PCT" | tr -d '%')
    if (( $(echo "$SPEED_PCT_NUM > 0" | bc -l) )); then
        echo "✓ Speed Improvement: ${SPEED_PCT}%"
    else
        echo "⚠ Speed Regression: ${SPEED_PCT}%"
    fi
fi

echo ""
echo "Detailed reports:"
echo "  $BASELINE_REPORT"
echo "  $OPTIMIZED_REPORT"
echo ""
