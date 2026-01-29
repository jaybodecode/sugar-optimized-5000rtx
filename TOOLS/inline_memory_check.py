#!/usr/bin/env python3
"""
GPU Memory Profiler - Inline version to run within active training process
Import this and call check_memory() during training without impact
"""

import torch
import gc
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


def check_memory(show_full_summary=False):
    """
    Quick memory check - safe to call during training
    Set show_full_summary=True for detailed breakdown
    """
    if not torch.cuda.is_available():
        console.print("[bold red]❌ CUDA not available[/bold red]")
        return
    
    # Basic stats
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
    mem_free, mem_total = torch.cuda.mem_get_info()
    mem_used = (mem_total - mem_free) / (1024**3)
    mem_total_gb = mem_total / (1024**3)
    fragmentation = reserved - allocated
    
    # Analyze tensors
    tensor_categories = defaultdict(float)
    tensor_dtypes = defaultdict(float)
    large_tensors = []
    total_tensors = 0
    
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                total_tensors += 1
                size_mb = obj.element_size() * obj.nelement() / (1024**2)
                
                # Categorize
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
                
                # Large tensors
                if size_mb > 50:
                    large_tensors.append({
                        'size_mb': size_mb,
                        'shape': tuple(obj.shape),
                        'dtype': dtype_name
                    })
        except:
            pass
    
    # Display quick overview
    table = Table(title="GPU Memory Snapshot", box=box.ROUNDED)
    table.add_column("Metric", style="cyan", width=25)
    table.add_column("Value", style="magenta", width=20)
    
    util_percent = (mem_used / mem_total_gb) * 100
    frag_percent = (fragmentation / reserved) * 100 if reserved > 0 else 0
    
    table.add_row("Total VRAM", f"{mem_total_gb:.2f} GB")
    table.add_row("Used", f"{mem_used:.2f} GB ({util_percent:.1f}%)")
    table.add_row("Allocated (PyTorch)", f"{allocated:.2f} GB")
    table.add_row("Reserved (PyTorch)", f"{reserved:.2f} GB")
    table.add_row("Fragmentation", f"{fragmentation:.2f} GB ({frag_percent:.1f}%)")
    table.add_row("Peak", f"{max_allocated:.2f} GB")
    table.add_row("---", "---")
    table.add_row("Total Tensors", f"{total_tensors:,}")
    
    console.print(table)
    
    # Category breakdown
    if tensor_categories:
        cat_table = Table(title="Tensor Categories", box=box.SIMPLE)
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column("Size (GB)", style="yellow", justify="right")
        cat_table.add_column("%", style="green", justify="right")
        
        total_mb = sum(tensor_categories.values())
        for cat, size_mb in sorted(tensor_categories.items(), key=lambda x: x[1], reverse=True):
            size_gb = size_mb / 1024
            pct = (size_mb / total_mb * 100) if total_mb > 0 else 0
            cat_table.add_row(cat.capitalize(), f"{size_gb:.3f}", f"{pct:.1f}%")
        
        console.print(cat_table)
    
    # DType breakdown
    if tensor_dtypes:
        dtype_table = Table(title="Data Types", box=box.SIMPLE)
        dtype_table.add_column("Type", style="cyan")
        dtype_table.add_column("Size (GB)", style="yellow", justify="right")
        
        for dtype, size_mb in sorted(tensor_dtypes.items(), key=lambda x: x[1], reverse=True):
            dtype_table.add_row(dtype, f"{size_mb/1024:.3f}")
        
        console.print(dtype_table)
    
    # Large tensors
    if large_tensors and show_full_summary:
        large_table = Table(title="Large Tensors (>50MB)", box=box.SIMPLE)
        large_table.add_column("#", width=3)
        large_table.add_column("Size (MB)", justify="right")
        large_table.add_column("Shape", style="dim")
        
        for idx, t in enumerate(sorted(large_tensors, key=lambda x: x['size_mb'], reverse=True)[:15], 1):
            large_table.add_row(str(idx), f"{t['size_mb']:.1f}", str(t['shape']))
        
        console.print(large_table)
    
    # Diagnostics
    diagnostics = []
    if frag_percent > 30:
        diagnostics.append(f"⚠️  High fragmentation ({frag_percent:.1f}%)")
    if util_percent > 90:
        diagnostics.append(f"🔴 Critical memory ({util_percent:.1f}%)")
    elif util_percent > 80:
        diagnostics.append(f"🟡 High memory ({util_percent:.1f}%)")
    else:
        diagnostics.append(f"🟢 Healthy memory ({util_percent:.1f}%)")
    
    if diagnostics:
        console.print(Panel("\n".join(diagnostics), title="Status", border_style="blue"))
    
    if show_full_summary:
        console.print("\n[dim]Full PyTorch Memory Summary:[/dim]")
        console.print(torch.cuda.memory_summary(abbreviated=True))


if __name__ == "__main__":
    check_memory(show_full_summary=True)
