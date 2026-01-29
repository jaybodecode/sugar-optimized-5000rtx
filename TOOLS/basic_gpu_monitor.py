#!/usr/bin/env python3
"""
Minimal GPU Monitor - No dependencies beyond PyTorch
"""

import torch
import time
import subprocess
import sys
from datetime import datetime

def clear_screen():
    print("\033[2J\033[H", end="")

def get_training_pid():
    """Find training process PID"""
    try:
        result = subprocess.run(['pgrep', '-f', 'train.py'], capture_output=True, text=True)
        if result.stdout:
            pids = result.stdout.strip().split('\n')
            # Return the one that's actually python training (not nsys wrapper)
            for pid in pids:
                check = subprocess.run(['ps', '-p', pid, '-o', 'cmd='], capture_output=True, text=True)
                if check.stdout and 'python' in check.stdout and 'train.py' in check.stdout:
                    return pid
            return pids[0] if pids else None
    except:
        pass
    return None

def get_process_gpu_memory(pid):
    """Get GPU memory for specific process using nvidia-smi"""
    if not pid:
        return None
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(',')
                if len(parts) >= 2 and parts[0].strip() == str(pid):
                    return float(parts[1].strip()) / 1024  # Convert MiB to GiB
    except:
        pass
    return None

def get_gpu_memory_nvidia_smi():
    """Get GPU memory directly from nvidia-smi (more reliable)"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if result.stdout:
            used_mb, total_mb = result.stdout.strip().split(',')
            return float(used_mb) / 1024, float(total_mb) / 1024  # Convert to GB
    except:
        pass
    return None, None

def main():
    interval = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    
    print("Starting GPU Monitor...")
    print("Press Ctrl+C to exit\n")
    time.sleep(1)
    
    try:
        while True:
            clear_screen()
            
            print(f"=" * 70)
            print(f"GPU MEMORY MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"=" * 70)
            print()
            
            # Get real GPU memory from nvidia-smi
            mem_used, mem_total_gb = get_gpu_memory_nvidia_smi()
            
            if mem_used is not None:
                util_pct = (mem_used / mem_total_gb) * 100
                
                print("GPU MEMORY:")
                print(f"  Total VRAM:       {mem_total_gb:6.2f} GB")
                print(f"  Used:             {mem_used:6.2f} GB ({util_pct:5.1f}%)")
                print(f"  Free:             {mem_total_gb - mem_used:6.2f} GB")
                
                # Status indicator
                if util_pct > 95:
                    status = "CRITICAL"
                elif util_pct > 80:
                    status = "HIGH"
                else:
                    status = "HEALTHY"
                print(f"\n  Status: {status}")
                
            else:
                print("GPU: NOT AVAILABLE")
            
            print()
            print("-" * 70)
            
            # Training Process
            pid = get_training_pid()
            if pid:
                proc_mem = get_process_gpu_memory(pid)
                print(f"TRAINING PROCESS: PID {pid} (RUNNING)")
                if proc_mem:
                    print(f"  Training GPU Memory: {proc_mem:6.2f} GB")
                    print(f"  Training Memory %:   {(proc_mem/mem_total_gb)*100:5.1f}%")
            else:
                print("TRAINING PROCESS: NOT FOUND")
            
            print("-" * 70)
            print(f"\nRefreshing every {interval}s... (Ctrl+C to exit)")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\nExiting...\n")
        sys.exit(0)

if __name__ == "__main__":
    main()
