#!/usr/bin/env python3
"""
Tensor Profile Log Analyzer
Generates summary reports from tensor profiling log files
"""

import os
import re
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def parse_log_file(log_path):
    """Parse tensor profile log file and extract metrics"""
    iterations = []
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Split by the header lines that precede each iteration
    blocks = re.split(r'={80,}', content)
    
    # Process blocks in pairs (header + data)
    i = 0
    while i < len(blocks):
        block = blocks[i]
        if not block.strip():
            i += 1
            continue
            
        # Extract iteration number from this block
        iter_match = re.search(r'Iteration (\d+)', block)
        if not iter_match:
            i += 1
            continue
        
        iteration = int(iter_match.group(1))
        
        # Data is in the next block (after the next ===)
        data_block = blocks[i+1] if i+1 < len(blocks) else ''
        full_block = block + data_block
        
        # Extract metrics
        data = {'iteration': iteration}
        
        # Total tensors
        if match := re.search(r'Total Tensors: (\d+)', full_block):
            data['total_tensors'] = int(match.group(1))
        
        # Memory stats
        if match := re.search(r'Allocated: ([\d.]+) MB', full_block):
            data['allocated_mb'] = float(match.group(1))
        if match := re.search(r'Reserved: ([\d.]+) MB', full_block):
            data['reserved_mb'] = float(match.group(1))
        if match := re.search(r'Fragmentation: ([\d.]+)%', full_block):
            data['fragmentation'] = float(match.group(1))
        
        # Categories
        data['categories'] = {}
        cat_section = re.search(r'Tensor Categories:(.*?)(?:Data Types:|$)', full_block, re.DOTALL)
        if cat_section:
            for line in cat_section.group(1).strip().split('\n'):
                if ':' in line and 'MB' in line:
                    parts = line.split(':')
                    if len(parts) == 2:
                        name = parts[0].strip()
                        value = float(parts[1].replace('MB', '').strip())
                        data['categories'][name] = value
        
        # Data types
        data['dtypes'] = {}
        dtype_section = re.search(r'Data Types:(.*?)(?:Large Tensors|$)', full_block, re.DOTALL)
        if dtype_section:
            for line in dtype_section.group(1).strip().split('\n'):
                if ':' in line and 'MB' in line:
                    parts = line.split(':')
                    if len(parts) == 2:
                        name = parts[0].strip()
                        value = float(parts[1].replace('MB', '').strip())
                        data['dtypes'][name] = value
        
        # Large tensors
        if match := re.search(r'Large Tensors \(>100MB\): (\d+)', full_block):
            data['large_tensors'] = int(match.group(1))
        
        iterations.append(data)
        i += 2  # Skip to next potential iteration block
    
    return iterations

def print_summary(iterations):
    """Print summary statistics"""
    if not iterations:
        print("No data found in log file")
        return
    
    print("\n" + "="*80)
    print("TENSOR PROFILING SUMMARY")
    print("="*80)
    
    first = iterations[0]
    last = iterations[-1]
    
    print(f"\nIteration Range: {first['iteration']} - {last['iteration']}")
    print(f"Total Snapshots: {len(iterations)}")
    
    # Memory trends
    print(f"\n{'Memory Allocated:':<25} Start: {first.get('allocated_mb', 0):.1f} MB  →  End: {last.get('allocated_mb', 0):.1f} MB")
    print(f"{'Memory Reserved:':<25} Start: {first.get('reserved_mb', 0):.1f} MB  →  End: {last.get('reserved_mb', 0):.1f} MB")
    print(f"{'Fragmentation:':<25} Start: {first.get('fragmentation', 0):.1f}%  →  End: {last.get('fragmentation', 0):.1f}%")
    print(f"{'Total Tensors:':<25} Start: {first.get('total_tensors', 0)}  →  End: {last.get('total_tensors', 0)}")
    
    # Category breakdown (last iteration)
    if last.get('categories'):
        print("\nTensor Categories (at end):")
        total_cat = sum(last['categories'].values())
        for name, size in sorted(last['categories'].items(), key=lambda x: x[1], reverse=True):
            pct = (size / total_cat * 100) if total_cat > 0 else 0
            print(f"  {name:15s}: {size:8.1f} MB  ({pct:5.1f}%)")
    
    # Dtype breakdown (last iteration)
    if last.get('dtypes'):
        print("\nData Types (at end):")
        total_dtype = sum(last['dtypes'].values())
        for name, size in sorted(last['dtypes'].items(), key=lambda x: x[1], reverse=True):
            pct = (size / total_dtype * 100) if total_dtype > 0 else 0
            print(f"  {name:15s}: {size:8.1f} MB  ({pct:5.1f}%)")
    
    # Large tensors
    print(f"\nLarge Tensors (>100MB): {last.get('large_tensors', 0)}")
    
    # Trend analysis
    print("\n" + "-"*80)
    print("TREND ANALYSIS")
    print("-"*80)
    
    if len(iterations) > 1:
        # Calculate deltas
        allocated_delta = last.get('allocated_mb', 0) - first.get('allocated_mb', 0)
        reserved_delta = last.get('reserved_mb', 0) - first.get('reserved_mb', 0)
        tensor_delta = last.get('total_tensors', 0) - first.get('total_tensors', 0)
        
        print(f"Memory Allocated Change: {allocated_delta:+.1f} MB")
        print(f"Memory Reserved Change:  {reserved_delta:+.1f} MB")
        print(f"Tensor Count Change:     {tensor_delta:+d}")
        
        # Check for memory leaks
        if allocated_delta > 100:
            print(f"\n⚠️  WARNING: Allocated memory increased by {allocated_delta:.1f} MB")
            print("   This may indicate a memory leak.")
        
        # Check fragmentation
        avg_frag = sum(it.get('fragmentation', 0) for it in iterations) / len(iterations)
        if avg_frag > 20:
            print(f"\n⚠️  WARNING: High average fragmentation ({avg_frag:.1f}%)")
            print("   Consider more frequent empty_cache() calls.")

def print_timeline(iterations, max_lines=20):
    """Print timeline of key metrics"""
    print("\n" + "="*80)
    print("MEMORY TIMELINE")
    print("="*80)
    print(f"{'Iter':>8} | {'Allocated':>10} | {'Reserved':>10} | {'Frag%':>6} | {'Tensors':>8}")
    print("-"*80)
    
    # Sample iterations if too many
    if len(iterations) > max_lines:
        step = len(iterations) // max_lines
        sampled = [iterations[i] for i in range(0, len(iterations), step)]
    else:
        sampled = iterations
    
    for it in sampled:
        print(f"{it['iteration']:8d} | "
              f"{it.get('allocated_mb', 0):8.1f} MB | "
              f"{it.get('reserved_mb', 0):8.1f} MB | "
              f"{it.get('fragmentation', 0):5.1f}% | "
              f"{it.get('total_tensors', 0):8d}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_tensor_profile.py <log_file>")
        print("\nExample:")
        print("  python analyze_tensor_profile.py ../output/profiles/tensor_profile_20260129_130000.log")
        print("\nOr analyze the latest log:")
        print("  python analyze_tensor_profile.py latest")
        sys.exit(1)
    
    log_path = sys.argv[1]
    
    # Handle 'latest' keyword
    if log_path == 'latest':
        # Find all tensor profile logs in common locations
        search_paths = [
            Path.cwd() / 'TOOLS' / 'profiles',
            Path.cwd() / 'profiles',
            Path.cwd().parent / 'SAMPLES',
            Path('/home/jason/GITHUB/SugarV3/SAMPLES'),
            Path('/home/jason/GITHUB/SugarV3/SuGaR/TOOLS/profiles')
        ]
        
        log_files = []
        for base_path in search_paths:
            if base_path.exists():
                log_files.extend(base_path.rglob('tensor_profile_*.log'))
        
        if not log_files:
            print("❌ No tensor profile logs found")
            print("\nSearched in:")
            for p in search_paths:
                print(f"  - {p}")
            print("\n💡 Logs are created when training runs with --profile_tensors True")
            print("   They will be in: <output_dir>/profiles/tensor_profile_*.log")
            sys.exit(1)
        
        # Get most recent
        log_path = max(log_files, key=lambda p: p.stat().st_mtime)
        print(f"📊 Analyzing latest log: {log_path}")
    
    if not os.path.exists(log_path):
        print(f"❌ Log file not found: {log_path}")
        sys.exit(1)
    
    print(f"📖 Reading: {log_path}")
    iterations = parse_log_file(log_path)
    
    if not iterations:
        print("❌ No iteration data found in log file")
        sys.exit(1)
    
    print_summary(iterations)
    print_timeline(iterations)
    
    print("\n" + "="*80)
    print(f"✅ Analysis complete ({len(iterations)} iterations)")
    print("="*80)

if __name__ == '__main__':
    main()
