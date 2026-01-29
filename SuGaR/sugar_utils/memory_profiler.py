"""
Tensor Memory Profiler for TensorBoard
Logs detailed GPU tensor breakdown during training
"""

import torch
import gc
import os
from datetime import datetime


def log_tensor_memory_breakdown(writer, iteration, interval=50, log_file=None):
    """
    Log detailed tensor memory breakdown to TensorBoard.
    Lightweight - runs every N iterations during logging phase.
    
    Args:
        writer: TensorBoard SummaryWriter
        iteration: Current training iteration
        interval: Log every N iterations (default 50)
    """
    if iteration % interval != 0:
        return
    
    # Categorize all GPU tensors
    tensor_categories = {
        'gradients': 0,
        'parameters': 0,
        'activations': 0,
        'buffers': 0,
        'other': 0
    }
    
    tensor_dtypes = {}
    large_tensors = []  # Track top memory consumers
    total_tensors = 0
    
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                total_tensors += 1
                size_mb = obj.element_size() * obj.nelement() / (1024**2)
                
                # Categorize (only check .grad on leaf tensors to avoid warning)
                if obj.grad_fn is not None or (hasattr(obj, 'requires_grad') and obj.requires_grad):
                    if hasattr(obj, 'is_leaf') and obj.is_leaf and obj.grad is not None:
                        tensor_categories['gradients'] += size_mb
                    else:
                        tensor_categories['activations'] += size_mb
                elif hasattr(obj, 'is_leaf') and obj.is_leaf:
                    tensor_categories['parameters'] += size_mb
                else:
                    tensor_categories['other'] += size_mb
                
                # Track by dtype
                dtype_name = str(obj.dtype).split('.')[-1]
                tensor_dtypes[dtype_name] = tensor_dtypes.get(dtype_name, 0) + size_mb
                
                # Track large tensors (>100MB)
                if size_mb > 100:
                    large_tensors.append({
                        'size_mb': size_mb,
                        'shape': tuple(obj.shape),
                        'dtype': dtype_name
                    })
        except:
            pass
    
    # Log to TensorBoard
    for category, size_mb in tensor_categories.items():
        writer.add_scalar(f'TensorProfiling/Tensor_Category/{category}_MB', size_mb, iteration)
    
    for dtype, size_mb in tensor_dtypes.items():
        writer.add_scalar(f'TensorProfiling/Tensor_DType/{dtype}_MB', size_mb, iteration)
    
    # Log count and total size of large tensors
    writer.add_scalar('TensorProfiling/Large_Tensors/count', len(large_tensors), iteration)
    total_large = sum(t['size_mb'] for t in large_tensors)
    writer.add_scalar('TensorProfiling/Large_Tensors/total_MB', total_large, iteration)
    
    # Total tensor count
    writer.add_scalar('TensorProfiling/Tensor_Count/total', total_tensors, iteration)
    
    # Fragmentation metric
    allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
    reserved = torch.cuda.memory_reserved() / (1024**2)
    fragmentation = (reserved - allocated) / reserved * 100 if reserved > 0 else 0
    writer.add_scalar('TensorProfiling/fragmentation_percent', fragmentation, iteration)
    
    # Also log to text file if provided
    if log_file:
        try:
            with open(log_file, 'a') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"\n{'='*80}\n")
                f.write(f"Iteration {iteration} | {timestamp}\n")
                f.write(f"{'='*80}\n")
                f.write(f"Total Tensors: {total_tensors}\n")
                f.write(f"Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB | Fragmentation: {fragmentation:.1f}%\n\n")
                f.write("Tensor Categories:\n")
                for cat, size in sorted(tensor_categories.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {cat:15s}: {size:8.2f} MB\n")
                f.write("\nData Types:\n")
                for dtype, size in sorted(tensor_dtypes.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {dtype:15s}: {size:8.2f} MB\n")
                f.write(f"\nLarge Tensors (>100MB): {len(large_tensors)}\n")
                if large_tensors:
                    f.write("Top 5 largest:\n")
                    for t in sorted(large_tensors, key=lambda x: x['size_mb'], reverse=True)[:5]:
                        f.write(f"  {t['size_mb']:8.2f} MB | shape={t['shape']} | dtype={t['dtype']}\n")
        except Exception as e:
            pass  # Don't crash training on logging error

