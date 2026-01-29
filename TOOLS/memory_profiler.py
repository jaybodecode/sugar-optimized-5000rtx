#!/usr/bin/env python3
"""
GPU Memory Profiler - Real-time VRAM analysis without impacting training
Provides detailed tensor-level breakdown and memory health diagnostics
"""

import torch
import gc
import sys
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich import box
import time

console = Console()


def get_memory_summary():
    """Get basic memory statistics"""
    if not torch.cuda.is_available():
        return None
    
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
    
    mem_free, mem_total = torch.cuda.mem_get_info()
    mem_used = (mem_total - mem_free) / (1024**3)
    mem_total_gb = mem_total / (1024**3)
    
    return {
        'allocated': allocated,
        'reserved': reserved,
        'max_allocated': max_allocated,
        'mem_used': mem_used,
        'mem_total': mem_total_gb,
        'fragmentation': reserved - allocated
    }


def analyze_tensors():
    """Analyze all GPU tensors in memory"""
    
    tensor_categories = defaultdict(float)
    tensor_dtypes = defaultdict(float)
    tensor_shapes = defaultdict(lambda: {'count': 0, 'total_mb': 0})
    large_tensors = []
    total_tensors = 0
    
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                total_tensors += 1
                size_mb = obj.element_size() * obj.nelement() / (1024**2)
                
                # Categorize by type
                if hasattr(obj, 'grad') and obj.grad is not None:
                    tensor_categories['gradients'] += size_mb
                elif hasattr(obj, 'requires_grad') and obj.requires_grad:
                    if hasattr(obj, 'is_leaf') and obj.is_leaf:
                        tensor_categories['parameters'] += size_mb
                    else:
                        tensor_categories['activations'] += size_mb
                else:
                    tensor_categories['buffers'] += size_mb
                
                # Track by dtype
                dtype_name = str(obj.dtype).split('.')[-1]
                tensor_dtypes[dtype_name] += size_mb
                
                # Track by shape pattern
                shape_str = f"{tuple(obj.shape)}"
                if len(shape_str) > 50:
                    shape_str = f"({obj.dim()}D tensor)"
                tensor_shapes[shape_str]['count'] += 1
                tensor_shapes[shape_str]['total_mb'] += size_mb
                
                # Track large tensors (>50MB)
                if size_mb > 50:
                    large_tensors.append({
                        'size_mb': size_mb,
                        'shape': tuple(obj.shape),
                        'dtype': dtype_name,
                        'requires_grad': getattr(obj, 'requires_grad', False)
                    })
        except Exception:
            pass
    
    return {
        'total_count': total_tensors,
        'categories': dict(tensor_categories),
        'dtypes': dict(tensor_dtypes),
        'shapes': dict(sorted(tensor_shapes.items(), key=lambda x: x[1]['total_mb'], reverse=True)[:10]),
        'large_tensors': sorted(large_tensors, key=lambda x: x['size_mb'], reverse=True)[:10]
    }


def create_memory_overview_table(mem_stats):
    """Create memory overview table"""
    table = Table(title="GPU Memory Overview", box=box.ROUNDED, show_header=True)
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Value", style="magenta", width=15)
    table.add_column("Status", style="green", width=30)
    
    # Memory utilization
    util_percent = (mem_stats['mem_used'] / mem_stats['mem_total']) * 100
    util_status = "🟢 Healthy" if util_percent < 80 else "🟡 High" if util_percent < 95 else "🔴 Critical"
    table.add_row("Total VRAM", f"{mem_stats['mem_total']:.2f} GB", "")
    table.add_row("Used", f"{mem_stats['mem_used']:.2f} GB", f"{util_percent:.1f}% {util_status}")
    table.add_row("Allocated", f"{mem_stats['allocated']:.2f} GB", "PyTorch tracked")
    table.add_row("Reserved", f"{mem_stats['reserved']:.2f} GB", "PyTorch cached")
    
    # Fragmentation
    frag_percent = (mem_stats['fragmentation'] / mem_stats['reserved']) * 100 if mem_stats['reserved'] > 0 else 0
    frag_status = "🟢 Low" if frag_percent < 20 else "🟡 Moderate" if frag_percent < 40 else "🔴 High"
    table.add_row("Fragmentation", f"{mem_stats['fragmentation']:.2f} GB", f"{frag_percent:.1f}% {frag_status}")
    
    table.add_row("Peak Usage", f"{mem_stats['max_allocated']:.2f} GB", "Since start")
    
    return table


def create_tensor_category_table(tensor_data):
    """Create tensor category breakdown table"""
    table = Table(title="Tensor Memory by Category", box=box.ROUNDED)
    table.add_column("Category", style="cyan", width=20)
    table.add_column("Size (MB)", style="yellow", justify="right", width=15)
    table.add_column("Size (GB)", style="magenta", justify="right", width=15)
    table.add_column("Percentage", style="green", justify="right", width=15)
    
    total_mb = sum(tensor_data['categories'].values())
    
    for category, size_mb in sorted(tensor_data['categories'].items(), key=lambda x: x[1], reverse=True):
        size_gb = size_mb / 1024
        percent = (size_mb / total_mb * 100) if total_mb > 0 else 0
        table.add_row(category.capitalize(), f"{size_mb:.1f}", f"{size_gb:.3f}", f"{percent:.1f}%")
    
    table.add_row("TOTAL", f"{total_mb:.1f}", f"{total_mb/1024:.3f}", "100.0%", style="bold")
    
    return table


def create_dtype_table(tensor_data):
    """Create dtype breakdown table"""
    table = Table(title="Memory by Data Type", box=box.ROUNDED)
    table.add_column("Data Type", style="cyan", width=20)
    table.add_column("Size (MB)", style="yellow", justify="right", width=15)
    table.add_column("Size (GB)", style="magenta", justify="right", width=15)
    table.add_column("Percentage", style="green", justify="right", width=15)
    
    total_mb = sum(tensor_data['dtypes'].values())
    
    for dtype, size_mb in sorted(tensor_data['dtypes'].items(), key=lambda x: x[1], reverse=True):
        size_gb = size_mb / 1024
        percent = (size_mb / total_mb * 100) if total_mb > 0 else 0
        table.add_row(dtype, f"{size_mb:.1f}", f"{size_gb:.3f}", f"{percent:.1f}%")
    
    if total_mb > 0:
        table.add_row("TOTAL", f"{total_mb:.1f}", f"{total_mb/1024:.3f}", "100.0%", style="bold")
    
    return table


def create_large_tensors_table(tensor_data):
    """Create large tensors table"""
    table = Table(title="Top 10 Largest Tensors", box=box.ROUNDED)
    table.add_column("#", style="dim", width=3)
    table.add_column("Size (MB)", style="yellow", justify="right", width=12)
    table.add_column("Shape", style="cyan", width=30)
    table.add_column("DType", style="magenta", width=12)
    table.add_column("Trainable", style="green", width=10)
    
    for idx, tensor_info in enumerate(tensor_data['large_tensors'], 1):
        shape_str = str(tensor_info['shape'])
        if len(shape_str) > 30:
            shape_str = shape_str[:27] + "..."
        
        trainable = "✓" if tensor_info['requires_grad'] else "✗"
        
        table.add_row(
            str(idx),
            f"{tensor_info['size_mb']:.1f}",
            shape_str,
            tensor_info['dtype'],
            trainable
        )
    
    return table


def create_shape_distribution_table(tensor_data):
    """Create shape distribution table"""
    table = Table(title="Top 10 Tensor Shapes (by total memory)", box=box.ROUNDED)
    table.add_column("Shape", style="cyan", width=35)
    table.add_column("Count", style="yellow", justify="right", width=10)
    table.add_column("Total (MB)", style="magenta", justify="right", width=12)
    table.add_column("Avg (MB)", style="green", justify="right", width=12)
    
    for shape_str, info in list(tensor_data['shapes'].items())[:10]:
        avg_mb = info['total_mb'] / info['count'] if info['count'] > 0 else 0
        table.add_row(
            shape_str,
            str(info['count']),
            f"{info['total_mb']:.1f}",
            f"{avg_mb:.1f}"
        )
    
    return table


def create_diagnostics_panel(mem_stats, tensor_data):
    """Create diagnostics and recommendations panel"""
    diagnostics = []
    
    # Check fragmentation
    frag_percent = (mem_stats['fragmentation'] / mem_stats['reserved']) * 100 if mem_stats['reserved'] > 0 else 0
    if frag_percent > 30:
        diagnostics.append(f"⚠️  High fragmentation ({frag_percent:.1f}%) - Consider torch.cuda.empty_cache()")
    else:
        diagnostics.append(f"✓ Fragmentation is acceptable ({frag_percent:.1f}%)")
    
    # Check memory pressure
    util_percent = (mem_stats['mem_used'] / mem_stats['mem_total']) * 100
    if util_percent > 90:
        diagnostics.append("⚠️  Very high memory utilization - Risk of OOM")
    elif util_percent > 80:
        diagnostics.append("⚠️  High memory utilization - Limited headroom")
    else:
        diagnostics.append(f"✓ Memory utilization is healthy ({util_percent:.1f}%)")
    
    # Check for gradient accumulation
    grad_mb = tensor_data['categories'].get('gradients', 0)
    if grad_mb > 2000:  # >2GB of gradients
        diagnostics.append(f"⚠️  Large gradient memory ({grad_mb:.0f} MB) - Possible accumulation issue")
    
    # Check dtype usage
    fp32_mb = tensor_data['dtypes'].get('float32', 0)
    fp16_mb = tensor_data['dtypes'].get('float16', 0) + tensor_data['dtypes'].get('bfloat16', 0)
    total_float = fp32_mb + fp16_mb
    if total_float > 0 and fp32_mb / total_float > 0.7:
        diagnostics.append(f"💡 {(fp32_mb/total_float*100):.0f}% FP32 tensors - Consider mixed precision")
    
    # Tensor count
    diagnostics.append(f"ℹ️  Total tensors: {tensor_data['total_count']:,}")
    
    text = "\n".join(diagnostics)
    return Panel(text, title="[bold]Diagnostics & Recommendations[/bold]", border_style="blue")


def run_analysis():
    """Run complete memory analysis"""
    console.print("\n[bold cyan]GPU Memory Profiler[/bold cyan]", justify="center")
    console.print("[dim]Analyzing GPU memory state...[/dim]\n", justify="center")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        console.print("[bold red]❌ CUDA not available[/bold red]")
        return
    
    # Get memory stats
    with console.status("[bold green]Collecting memory statistics..."):
        mem_stats = get_memory_summary()
    
    with console.status("[bold green]Analyzing tensors..."):
        tensor_data = analyze_tensors()
    
    # Display results
    console.print(create_memory_overview_table(mem_stats))
    console.print()
    
    console.print(create_tensor_category_table(tensor_data))
    console.print()
    
    console.print(create_dtype_table(tensor_data))
    console.print()
    
    if tensor_data['large_tensors']:
        console.print(create_large_tensors_table(tensor_data))
        console.print()
    
    console.print(create_shape_distribution_table(tensor_data))
    console.print()
    
    console.print(create_diagnostics_panel(mem_stats, tensor_data))
    console.print()
    
    # Print detailed summary
    console.print(Panel.fit(
        torch.cuda.memory_summary(device=0, abbreviated=False),
        title="[bold]PyTorch Memory Summary[/bold]",
        border_style="dim"
    ))


if __name__ == "__main__":
    try:
        run_analysis()
    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis interrupted[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)
