#!/bin/bash
# Enhanced VRAM Monitor - Tracks VmSize (ALL allocations including CUDA driver mappings)
# Usage: bash monitor_vram_v2.sh [logfile]
# Example: bash monitor_vram_v2.sh benchmark_logs/phase2_vram.csv

INTERVAL=5
LOGFILE="${1:-}"

echo "========================================"
echo "Enhanced System Monitor - Sampling every ${INTERVAL}s"
echo "Tracking: VRAM, Process RSS (physical), VmSize (all allocations + CUDA)"
echo "VmSize Delta shows CUDA memory region allocations"
echo "Press Ctrl+C to stop"
echo "========================================"
echo ""

# Find the training process PID (exclude conda wrapper)
find_training_pid() {
    ps aux | grep -E "python train\.py" | grep -v conda | grep -v grep | awk '{print $2}' | head -1
}

PYTHON_PID=$(find_training_pid)
if [ -z "$PYTHON_PID" ]; then
    echo "⚠ Warning: train.py process not found yet. Will search periodically..."
fi

# Check if iostat is available
if ! command -v iostat &> /dev/null; then
    echo "⚠ Warning: iostat not found (install sysstat for disk I/O monitoring)"
    DISK_IO_AVAILABLE=false
else
    DISK_IO_AVAILABLE=true
fi

# Print header
if [ "$DISK_IO_AVAILABLE" = true ]; then
    printf "%-19s | %8s | %5s | %5s | %9s | %9s | %8s | %8s | %8s | %8s\n" \
        "Timestamp" "VRAM" "GPU%" "CPU%" "Proc RSS" "Proc VmS" "Delta MB" "Sys RAM" "Read MB/s" "Write MB/s"
    printf "%-19s-|-%8s-|-%5s-|-%5s-|-%9s-|-%9s-|-%8s-|-%8s-|-%9s-|-%11s\n" \
        "-------------------" "--------" "-----" "-----" "---------" "---------" "--------" "--------" "---------" "-----------"
else
    printf "%-19s | %8s | %5s | %5s | %9s | %9s | %8s | %8s\n" \
        "Timestamp" "VRAM" "GPU%" "CPU%" "Proc RSS" "Proc VmS" "Delta MB" "Sys RAM"
    printf "%-19s-|-%8s-|-%5s-|-%5s-|-%9s-|-%9s-|-%8s-|-%8s\n" \
        "-------------------" "--------" "-----" "-----" "---------" "---------" "--------" "--------"
fi

# Initialize
PREV_VMSIZE_MB=0

# Write CSV header if logging
if [ -n "$LOGFILE" ]; then
    if [ "$DISK_IO_AVAILABLE" = true ]; then
        echo "Timestamp,VRAM_GB,GPU_Util%,CPU%,Process_RSS_GB,Process_VmSize_GB,VmSize_Delta_MB,System_RAM_GB,Disk_Read_MB/s,Disk_Write_MB/s" > "$LOGFILE"
    else
        echo "Timestamp,VRAM_GB,GPU_Util%,CPU%,Process_RSS_GB,Process_VmSize_GB,VmSize_Delta_MB,System_RAM_GB" > "$LOGFILE"
    fi
fi

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Get GPU stats
    GPU_STATS=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    VRAM_MB=$(echo "$GPU_STATS" | awk -F, '{print $1}' | xargs)
    GPU_UTIL=$(echo "$GPU_STATS" | awk -F, '{print $2}' | xargs)
    VRAM_GB=$(echo "scale=2; $VRAM_MB / 1024" | bc)
    
    # Get CPU usage
    CPU_UTIL=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    
    # Find process if not set
    if [ -z "$PYTHON_PID" ] || ! kill -0 "$PYTHON_PID" 2>/dev/null; then
        PYTHON_PID=$(find_training_pid)
    fi
    
    # Get process memory (RSS = physical, VmSize = all allocations including CUDA)
    if [ -n "$PYTHON_PID" ] && kill -0 "$PYTHON_PID" 2>/dev/null; then
        if [ -f "/proc/$PYTHON_PID/status" ]; then
            VMRSS_KB=$(grep "VmRSS:" /proc/$PYTHON_PID/status | awk '{print $2}')
            VMSIZE_KB=$(grep "VmSize:" /proc/$PYTHON_PID/status | awk '{print $2}')
            
            PROC_RSS_MB=$(echo "scale=0; $VMRSS_KB / 1024" | bc)
            PROC_RSS_GB=$(echo "scale=2; $PROC_RSS_MB / 1024" | bc)
            
            PROC_VMSIZE_MB=$(echo "scale=0; $VMSIZE_KB / 1024" | bc)
            PROC_VMSIZE_GB=$(echo "scale=2; $PROC_VMSIZE_MB / 1024" | bc)
        else
            # Fallback
            PROC_RSS_GB="N/A"
            PROC_VMSIZE_GB="N/A"
            VMSIZE_DELTA_MB=0
        fi
        
        # Calculate VmSize delta (tracks CUDA allocations)
        if [ "$PROC_VMSIZE_GB" != "N/A" ] && [ "$PREV_VMSIZE_MB" -ne 0 ]; then
            VMSIZE_DELTA_MB=$((PROC_VMSIZE_MB - PREV_VMSIZE_MB))
        else
            VMSIZE_DELTA_MB=0
        fi
        if [ "$PROC_VMSIZE_GB" != "N/A" ]; then
            PREV_VMSIZE_MB=$PROC_VMSIZE_MB
        fi
    else
        PROC_RSS_GB="N/A"
        PROC_VMSIZE_GB="N/A"
        VMSIZE_DELTA_MB=0
    fi
    
    # Get system RAM
    RAM_STATS=$(free -m | grep Mem)
    RAM_USED=$(echo "$RAM_STATS" | awk '{print $3}')
    SYS_RAM_GB=$(echo "scale=1; $RAM_USED / 1024" | bc)
    
    # Get disk I/O if available
    if [ "$DISK_IO_AVAILABLE" = true ]; then
        DISK_STATS=$(iostat -dxm 1 2 | tail -n +4 | grep -E "^(sd|nvme)" | tail -n +2)
        DISK_READ=$(echo "$DISK_STATS" | awk '{sum+=$4} END {printf "%.2f", sum}')
        DISK_WRITE=$(echo "$DISK_STATS" | awk '{sum+=$10} END {printf "%.2f", sum}')
    fi
    
    # Format delta
    if [ "$VMSIZE_DELTA_MB" -gt 0 ]; then
        DELTA_STR=$(printf "+%7d" "$VMSIZE_DELTA_MB")
    elif [ "$VMSIZE_DELTA_MB" -lt 0 ]; then
        DELTA_STR=$(printf "%8d" "$VMSIZE_DELTA_MB")
    else
        DELTA_STR="       0"
    fi
    
    # Print to terminal
    if [ "$DISK_IO_AVAILABLE" = true ]; then
        if [ "$PROC_RSS_GB" = "N/A" ]; then
            printf "%-19s | %6.2f GB | %4s%% | %4.0f%% |       N/A |       N/A |      N/A | %6.1f GB |     %5s |      %5s\n" \
                "$TIMESTAMP" "$VRAM_GB" "$GPU_UTIL" "$CPU_UTIL" "$SYS_RAM_GB" "$DISK_READ" "$DISK_WRITE"
        else
            printf "%-19s | %6.2f GB | %4s%% | %4.0f%% | %7.2f GB | %7.2f GB | %8s | %6.1f GB |     %5s |      %5s\n" \
                "$TIMESTAMP" "$VRAM_GB" "$GPU_UTIL" "$CPU_UTIL" "$PROC_RSS_GB" "$PROC_VMSIZE_GB" "$DELTA_STR" "$SYS_RAM_GB" "$DISK_READ" "$DISK_WRITE"
        fi
    else
        if [ "$PROC_RSS_GB" = "N/A" ]; then
            printf "%-19s | %6.2f GB | %4s%% | %4.0f%% |       N/A |       N/A |      N/A | %6.1f GB\n" \
                "$TIMESTAMP" "$VRAM_GB" "$GPU_UTIL" "$CPU_UTIL" "$SYS_RAM_GB"
        else
            printf "%-19s | %6.2f GB | %4s%% | %4.0f%% | %7.2f GB | %7.2f GB | %8s | %6.1f GB\n" \
                "$TIMESTAMP" "$VRAM_GB" "$GPU_UTIL" "$CPU_UTIL" "$PROC_RSS_GB" "$PROC_VMSIZE_GB" "$DELTA_STR" "$SYS_RAM_GB"
        fi
    fi
    
    # Log to file
    if [ -n "$LOGFILE" ] && [ "$PROC_RSS_GB" != "N/A" ]; then
        if [ "$DISK_IO_AVAILABLE" = true ]; then
            echo "$TIMESTAMP,$VRAM_GB,$GPU_UTIL,$CPU_UTIL,$PROC_RSS_GB,$PROC_VMSIZE_GB,$VMSIZE_DELTA_MB,$SYS_RAM_GB,$DISK_READ,$DISK_WRITE" >> "$LOGFILE"
        else
            echo "$TIMESTAMP,$VRAM_GB,$GPU_UTIL,$CPU_UTIL,$PROC_RSS_GB,$PROC_VMSIZE_GB,$VMSIZE_DELTA_MB,$SYS_RAM_GB" >> "$LOGFILE"
        fi
    fi
    
    sleep $INTERVAL
done
