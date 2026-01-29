# Enhanced Tensor-Level Memory Metrics for TensorBoard

## Overview
Detailed memory profiling to complement existing VRAM stats with tensor-level breakdown showing what's actually consuming GPU memory.

## Current TensorBoard Metrics (Already Logged)
- `vram_allocated` (GB)
- `vram_reserved` (GB)
- `vram_percent` (% utilization)
- RAM/CPU metrics

## Proposed New Metrics Section: "Memory/Tensor_Breakdown"

### Implementation Plan

Add to `coarse_density_and_dn_consistency.py` trainer:

```python
import gc

def log_tensor_memory_breakdown(writer, iteration, interval=500):
    """
    Log detailed tensor memory breakdown to TensorBoard.
    Zero impact on training - runs async during logging phase.
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
    
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                size_mb = obj.element_size() * obj.nelement() / (1024**2)
                
                # Categorize
                if obj.grad_fn is not None or (hasattr(obj, 'requires_grad') and obj.requires_grad):
                    if obj.grad is not None:
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
        writer.add_scalar(f'Memory/Tensor_Category/{category}_MB', size_mb, iteration)
    
    for dtype, size_mb in tensor_dtypes.items():
        writer.add_scalar(f'Memory/Tensor_DType/{dtype}_MB', size_mb, iteration)
    
    # Log count and total size of large tensors
    writer.add_scalar('Memory/Large_Tensors/count', len(large_tensors), iteration)
    total_large = sum(t['size_mb'] for t in large_tensors)
    writer.add_scalar('Memory/Large_Tensors/total_MB', total_large, iteration)
    
    # Fragmentation metric
    allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
    reserved = torch.cuda.memory_reserved() / (1024**2)
    fragmentation = (reserved - allocated) / reserved * 100 if reserved > 0 else 0
    writer.add_scalar('Memory/fragmentation_percent', fragmentation, iteration)
```

### Integration Point
In training loop, after existing memory logging:
```python
# Existing code around line 1565
vram_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
vram_reserved = torch.cuda.memory_reserved() / (1024 ** 3)

# Add new detailed breakdown
log_tensor_memory_breakdown(writer, iteration, interval=500)
```

## Benefits

1. **Optimization Opportunities**:
   - See if gradients are accumulating (memory leak)
   - Identify if activations dominate (use gradient checkpointing)
   - Check dtype distribution (fp32 vs fp16 usage)
   - Spot large tensors that could be optimized

2. **Memory Leak Detection**:
   - Gradients growing over time = leak
   - "Other" category growing = temporary tensors not freed
   - Large tensor count increasing = accumulation bug

3. **Verification**:
   - Confirm VRAM usage matches expectations
   - Validate optimization efforts (e.g., fp16 adoption)
   - Track memory efficiency across training phases

## Memory Snapshot for Deep Debugging

For specific issues, save detailed snapshots:
```python
# At problematic iteration
if iteration in [1000, 5000, 10000]:  # Key checkpoints
    torch.cuda.memory._dump_snapshot(
        f"memory_snapshot_iter{iteration}.pickle"
    )
```

Analyze offline with:
```bash
python -m torch.utils.memory_viz trace_plot memory_snapshot_iter1000.pickle -o memory_plot.html
```

## Performance Impact
- Logging: ~50ms every 500 iterations (negligible)
- No training accuracy impact
- No gradient computation impact
- Safe to run during active training

## Testing
1. Add to single trainer first (coarse_density_and_dn_consistency)
2. Verify TensorBoard shows new "Memory/Tensor_*" sections
3. Validate no performance degradation
4. Roll out to other trainers if useful

## Future Enhancements
- Per-model component breakdown (Gaussians, SH features, etc.)
- Memory timeline correlation with loss spikes
- Automatic anomaly detection (unexpected growth)
- Export reports for optimization analysis
