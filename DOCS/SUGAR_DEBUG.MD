# SuGaR Debugging & Performance Analysis Guide

Comprehensive guide for debugging, profiling, and analyzing GPU memory usage during SuGaR training.

## Table of Contents
- [Tensor Memory Profiling](#tensor-memory-profiling)
- [GPU Monitoring Tools](#gpu-monitoring-tools)
- [TensorBoard Integration](#tensorboard-integration)
- [Analyzing Results](#analyzing-results)
- [Troubleshooting](#troubleshooting)

---

## Tensor Memory Profiling

### Overview
SuGaR includes built-in tensor memory profiling to understand GPU VRAM composition during training. This helps identify:
- Memory leaks
- Large tensor allocations
- Fragmentation issues
- Category breakdown (parameters, gradients, activations)

### Enabling Profiling

Add the `--profile_tensors True` flag to your training command:

```bash
python train.py \
  -s data/your_scene \
  -c checkpoints/your_checkpoint \
  -i 7000 \
  -r density \
  --profile_tensors True
```

**Performance Impact:** ~50-200ms overhead every 50 iterations (negligible)

### What Gets Logged

Every 50 iterations, the profiler captures:
- **Tensor Categories:**
  - `parameters`: Model weights (_points, _scales, _quaternions, etc.)
  - `activations`: Intermediate computation tensors
  - `gradients`: Gradient tensors for backpropagation
  - `buffers`: Registered buffers
  - `other`: Uncategorized CUDA tensors

- **Data Types:** Memory by dtype (float32, float16, int64, etc.)
- **Large Tensors:** Count and details of tensors >100MB
- **Fragmentation:** Reserved vs allocated memory ratio

### Output Locations

1. **TensorBoard Graphs:** `<output_dir>/tensorboard/`
   - Navigate to "TensorProfiling" section
   - Real-time graphs during training

2. **Text Logs:** `<output_dir>/profiles/tensor_profile_TIMESTAMP.log`
   - Human-readable snapshot format
   - Suitable for post-training analysis

---

## GPU Monitoring Tools

### Basic GPU Monitor

Real-time GPU memory monitoring in a separate terminal:

```bash
# From SuGaR directory
python TOOLS/basic_gpu_monitor.py
```

**Features:**
- Updates every 2 seconds
- Shows total GPU memory usage
- Uses nvidia-smi (works across processes)
- Minimal CPU overhead

**Output:**
```
GPU Memory: 15.6 GB / 16.0 GB (97%)
```

### NVIDIA Nsight Systems Profiling

For detailed GPU memory timeline and CUDA kernel analysis:

```bash
# Capture 30-second profile of running training
./TOOLS/nsight_snapshot.sh

# Full training session profile
./TOOLS/nsight_profile_training.sh
```

**Analysis:**
```bash
# View summary
nsys stats --report cuda_gpu_mem_size_sum profile.nsys-rep

# Open GUI
nsys-ui profile.nsys-rep
```

---

## TensorBoard Integration

### Starting TensorBoard

```bash
# View current experiment
tensorboard --logdir "output/your_experiment/tensorboard" --port 6007 --bind_all

# Compare multiple experiments (if using _mesh naming)
tensorboard --logdir "output/scene_mesh/" --port 6007 --bind_all
```

Access at: `http://localhost:6007`

### TensorProfiling Metrics

Navigate to the **TensorProfiling** tab to view:

| Metric | Description |
|--------|-------------|
| `Tensor_Category/*_MB` | Memory by category (parameters/activations/gradients) |
| `Tensor_DType/*_MB` | Memory by data type (float32/float16/int64) |
| `Large_Tensors/count` | Number of tensors >100MB |
| `Large_Tensors/total_MB` | Total memory in large tensors |
| `Tensor_Count/total` | Total CUDA tensor count |
| `fragmentation_percent` | Memory fragmentation (reserved - allocated) |

### Interpreting Graphs

**Normal Patterns:**
- Parameters: Stable (only changes during densification/pruning)
- Activations: Fluctuates with batch processing
- Gradients: Stable after optimizer warmup
- Fragmentation: 10-30% is normal for Gaussian Splatting

**Warning Signs:**
- Parameters continuously growing → memory leak in model
- Activations steadily increasing → accumulating intermediate tensors
- Fragmentation >40% → consider `torch.cuda.empty_cache()` calls

---

## Analyzing Results

### Post-Training Analysis

Use the built-in analyzer to generate reports from log files:

```bash
# Analyze latest profiling log
python TOOLS/analyze_tensor_profile.py latest

# Or specify a file
python TOOLS/analyze_tensor_profile.py output/profiles/tensor_profile_20260129_133621.log
```

### Sample Report

```
================================================================================
TENSOR PROFILING SUMMARY
================================================================================

Iteration Range: 40000 - 50000
Total Snapshots: 201

Memory Allocated:         Start: 8073.4 MB  →  End: 8156.2 MB
Memory Reserved:          Start: 10126.0 MB  →  End: 10976.0 MB
Fragmentation:            Start: 20.3%  →  End: 25.8%
Total Tensors:            Start: 1267  →  End: 1289

Tensor Categories (at end):
  parameters     :   5224.7 MB  ( 64.1%)
  activations    :   2860.4 MB  ( 35.1%)
  gradients      :     68.7 MB  (  0.8%)

Data Types (at end):
  float32        :   7549.9 MB  ( 92.6%)
  int64          :    575.0 MB  (  7.1%)

Large Tensors (>100MB): 7
```

### Understanding the Numbers

**Memory Allocated vs Reserved:**
- **Allocated:** Actual memory in use by tensors
- **Reserved:** Memory pool held by PyTorch allocator
- **Difference:** Fragmented space available for reuse

**Fragmentation Analysis:**
```
20-30%: Normal for Gaussian Splatting (many small tensors)
30-40%: Moderate - consider periodic empty_cache()
>40%:   High - add torch.cuda.empty_cache() calls
```

**Large Tensors:**
Most common large allocations in SuGaR:
- `(N, 15, 3)` - Spherical harmonics rest coefficients (~1GB per 1M Gaussians)
- `(N, 3)` - Point positions, scales, DC coefficients
- `(N, 4)` - Quaternion rotations

---

## Troubleshooting

### Out of Memory Errors

**Check fragmentation first:**
```bash
python TOOLS/analyze_tensor_profile.py latest
```

If fragmentation >40%, add to training code:
```python
if iteration % 100 == 0:
    torch.cuda.empty_cache()
```

**Reduce memory usage:**
- Train at half resolution initially
- Reduce point count (lower densification thresholds)
- Use `--high_poly False` for refinement
- Consider gradient checkpointing for large models

### Profiling Not Working

**Symptoms:**
- No `profiles/` directory created
- Empty log files
- No TensorProfiling section in TensorBoard

**Solutions:**
1. Verify flag is set: `--profile_tensors True` (not `--profile_tensors` alone)
2. Check training started from correct iteration (profiling begins after loading)
3. Wait for first profiling iteration (runs every 50 iterations)

### nvidia-smi Shows [N/A] for Process Memory

This is a driver limitation on newer GPUs (RTX 40/50 series with driver 590+). Solutions:
- Use total GPU memory from nvidia-smi (works correctly)
- Use built-in tensor profiler (process-internal, accurate)
- Use nsys profiling for detailed timeline

### TensorBoard Graphs Not Updating

```bash
# Kill existing TensorBoard
pkill -f tensorboard

# Restart with fresh cache
tensorboard --logdir "output/experiment/tensorboard" --port 6007 --bind_all --reload_interval 5
```

---

## Advanced Usage

### Custom Profiling Intervals

Edit `sugar_trainers/coarse_density_and_dn_consistency.py`:
```python
# Change interval from 50 to your desired value
log_tensor_memory_breakdown(tb_writer, iteration, interval=100, log_file=tensor_profile_log)
```

### Profiling Specific Phases

Add conditional profiling:
```python
if iteration >= 45000 and iteration <= 48000:  # Profile critical phase only
    if hasattr(args, 'profile_tensors') and args.profile_tensors:
        log_tensor_memory_breakdown(tb_writer, iteration, interval=10)
```

### Export for External Analysis

Logs are plain text - use standard tools:
```bash
# Extract fragmentation values
grep "Fragmentation:" tensor_profile_*.log

# Plot with gnuplot, matplotlib, etc.
python -c "
import re
with open('tensor_profile.log') as f:
    for line in f:
        if 'Allocated:' in line:
            mb = re.search(r'Allocated: ([\d.]+) MB', line)
            print(mb.group(1))
"
```

---

## Best Practices

1. **Enable profiling for baseline runs** - Understand normal memory patterns
2. **Compare experiments** - Use TensorBoard multi-run comparison
3. **Profile before OOM** - Catch issues early in development
4. **Archive logs** - Keep profiling logs with model checkpoints
5. **Monitor trends** - Watch for gradual memory increases over training

---

## Tool Reference

### Created Files

| File | Purpose |
|------|---------|
| `TOOLS/basic_gpu_monitor.py` | Real-time GPU memory monitor |
| `TOOLS/nsight_snapshot.sh` | Capture nsys profile snapshot |
| `TOOLS/analyze_tensor_profile.py` | Parse and analyze profiling logs |
| `sugar_utils/memory_profiler.py` | Core profiling module |

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--profile_tensors` | False | Enable tensor memory profiling |
| `--checkpoint_interval` | 2000 | Save checkpoints (also good for debugging) |
| `--eval` | False | Enable test set evaluation (adds LPIPS overhead) |

---

## FAQ

**Q: Does profiling slow down training?**  
A: Minimal impact - ~50-200ms every 50 iterations. 99% of time is unchanged.

**Q: Can I profile during refinement?**  
A: Yes, but refinement trainer needs modification (currently only in coarse trainer).

**Q: How much disk space do logs use?**  
A: ~1KB per snapshot. Full 10K iteration run ≈ 200KB.

**Q: Can I profile multi-GPU training?**  
A: Currently single GPU only. Multi-GPU support requires DDP-aware profiling.

**Q: Why are my "parameters" higher than model size?**  
A: Profiler counts optimizer state (momentum, Adam buffers). This is normal.

---

## Related Documentation

- [VRAM_OPTIMIZATION.md](VRAM_OPTIMIZATION.md) - Memory reduction techniques
- [SUGAR_TRAIN.MD](SUGAR_TRAIN.MD) - Training guide
- [MIPS_USAGE.MD](MIPS_USAGE.MD) - MIPS-Splatting debugging

---

*Last Updated: January 2026*
