#!/usr/bin/env python3
"""
Live GPU Memory Monitor - Run in separate terminal while training
Real-time dashboard showing GPU memory usage, tensor counts, and training process info
"""

import torch
import gc
import psutil
import time
import sys
from datetime import datetime, timedelta
from collections import defaultdict, deque
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich import box
from rich.text import Text

console = Console()


class GPUMonitor:
    def __init__(self, update_interval=2.0):
        self.update_interval = update_interval
        self.history_length = 30
        self.vram_history = deque(maxlen=self.history_length)
        self.start_time = time.time()
        
    def find_training_process(self):
        """Find active training process"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if cmdline and any('train.py' in str(arg) for arg in cmdline):
                    return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return None
    
    def get_gpu_stats(self):
        """Get current GPU statistics"""
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
            'fragmentation': reserved - allocated,
            'utilization': (mem_used / mem_total_gb) * 100
        }
    
    def analyze_tensors_quick(self):
        """Quick tensor analysis without full GC scan"""
        tensor_count = 0
        total_size_mb = 0
        dtypes = defaultdict(int)
        
        # Sample only - don't scan everything (too slow)
        sample_size = 0
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    tensor_count += 1
                    size_mb = obj.element_size() * obj.nelement() / (1024**2)
                    total_size_mb += size_mb
                    dtype = str(obj.dtype).split('.')[-1]
                    dtypes[dtype] += 1
                    
                    sample_size += 1
                    if sample_size > 1000:  # Limit scan for performance
                        break
            except:
                pass
        
        return {
            'count': tensor_count,
            'total_mb': total_size_mb,
            'dtypes': dict(dtypes)
        }
    
    def get_process_info(self, proc):
        """Get training process information"""
        try:
            with proc.oneshot():
                cpu_percent = proc.cpu_percent()
                mem_info = proc.memory_info()
                ram_gb = mem_info.rss / (1024**3)
                
                # Get command line
                cmdline = ' '.join(proc.cmdline())
                if len(cmdline) > 100:
                    cmdline = cmdline[:97] + "..."
                
                return {
                    'pid': proc.pid,
                    'cpu_percent': cpu_percent,
                    'ram_gb': ram_gb,
                    'cmdline': cmdline,
                    'status': proc.status()
                }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None
    
    def create_gpu_table(self, gpu_stats):
        """Create GPU statistics table"""
        table = Table(title="🎮 GPU Memory", box=box.ROUNDED, show_header=False)
        table.add_column("Metric", style="cyan", width=25)
        table.add_column("Value", style="magenta", width=25)
        
        util_color = "green" if gpu_stats['utilization'] < 80 else "yellow" if gpu_stats['utilization'] < 95 else "red"
        
        table.add_row("Total VRAM", f"{gpu_stats['mem_total']:.2f} GB")
        table.add_row("Used", f"[{util_color}]{gpu_stats['mem_used']:.2f} GB ({gpu_stats['utilization']:.1f}%)[/{util_color}]")
        table.add_row("PyTorch Allocated", f"{gpu_stats['allocated']:.2f} GB")
        table.add_row("PyTorch Reserved", f"{gpu_stats['reserved']:.2f} GB")
        
        frag_pct = (gpu_stats['fragmentation'] / gpu_stats['reserved'] * 100) if gpu_stats['reserved'] > 0 else 0
        frag_color = "green" if frag_pct < 20 else "yellow" if frag_pct < 40 else "red"
        table.add_row("Fragmentation", f"[{frag_color}]{gpu_stats['fragmentation']:.2f} GB ({frag_pct:.1f}%)[/{frag_color}]")
        
        table.add_row("Peak Usage", f"{gpu_stats['max_allocated']:.2f} GB")
        
        return table
    
    def create_process_table(self, proc_info):
        """Create training process table"""
        table = Table(title="🚀 Training Process", box=box.ROUNDED, show_header=False)
        table.add_column("Metric", style="cyan", width=25)
        table.add_column("Value", style="yellow", width=25)
        
        if proc_info:
            table.add_row("PID", str(proc_info['pid']))
            table.add_row("Status", proc_info['status'].upper())
            table.add_row("CPU Usage", f"{proc_info['cpu_percent']:.1f}%")
            table.add_row("RAM Usage", f"{proc_info['ram_gb']:.2f} GB")
        else:
            table.add_row("Status", "[red]NO TRAINING FOUND[/red]")
        
        return table
    
    def create_tensor_table(self, tensor_info):
        """Create tensor statistics table"""
        table = Table(title="📊 Tensor Info", box=box.ROUNDED, show_header=False)
        table.add_column("Metric", style="cyan", width=25)
        table.add_column("Value", style="green", width=25)
        
        table.add_row("Total Tensors", f"{tensor_info['count']:,}")
        table.add_row("Total Size", f"{tensor_info['total_mb']/1024:.2f} GB")
        
        if tensor_info['dtypes']:
            dtype_str = ", ".join([f"{k}:{v}" for k, v in sorted(tensor_info['dtypes'].items(), key=lambda x: x[1], reverse=True)[:3]])
            table.add_row("Top DTypes", dtype_str)
        
        return table
    
    def create_sparkline(self, data, width=40):
        """Create ASCII sparkline from data"""
        if len(data) < 2:
            return "─" * width
        
        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val if max_val > min_val else 1
        
        chars = "▁▂▃▄▅▆▇█"
        sparkline = ""
        
        for val in data:
            normalized = (val - min_val) / range_val
            char_idx = min(int(normalized * (len(chars) - 1)), len(chars) - 1)
            sparkline += chars[char_idx]
        
        return sparkline
    
    def create_history_panel(self):
        """Create VRAM history sparkline"""
        if len(self.vram_history) > 1:
            sparkline = self.create_sparkline(list(self.vram_history), width=50)
            latest = self.vram_history[-1] if self.vram_history else 0
            text = f"VRAM Usage History (last {len(self.vram_history)} samples)\n"
            text += f"{sparkline}\n"
            text += f"Current: {latest:.2f} GB"
        else:
            text = "Collecting history..."
        
        return Panel(text, title="📈 History", border_style="blue")
    
    def create_layout(self, gpu_stats, proc_info, tensor_info):
        """Create dashboard layout"""
        from rich.console import Group
        
        # Header
        uptime = str(timedelta(seconds=int(time.time() - self.start_time)))
        header = Panel(
            f"[bold cyan]GPU Memory Monitor[/bold cyan] | Uptime: {uptime} | {datetime.now().strftime('%H:%M:%S')}",
            style="bold white on blue"
        )
        
        # Tables
        gpu_table = self.create_gpu_table(gpu_stats)
        proc_table = self.create_process_table(proc_info)
        tensor_table = self.create_tensor_table(tensor_info)
        history_panel = self.create_history_panel()
        
        # Command info
        cmd_panel = Panel(
            proc_info['cmdline'] if proc_info else "[red]No training process found[/red]",
            title="Command",
            border_style="dim"
        )
        
        # Return Group of renderables
        return Group(
            header,
            Text(""),
            gpu_table,
            Text(""),
            proc_table,
            Text(""),
            tensor_table,
            Text(""),
            history_panel,
            Text(""),
            cmd_panel
        )
    
    def run(self):
        """Run live monitoring dashboard"""
        console.print("[bold green]Starting GPU Memory Monitor...[/bold green]")
        console.print("[dim]Press Ctrl+C to stop[/dim]\n")
        
        with Live(console=console, refresh_per_second=0.5, screen=True) as live:
            while True:
                try:
                    # Collect data
                    gpu_stats = self.get_gpu_stats()
                    if not gpu_stats:
                        live.update("[bold red]CUDA not available[/bold red]")
                        time.sleep(self.update_interval)
                        continue
                    
                    proc = self.find_training_process()
                    proc_info = self.get_process_info(proc) if proc else None
                    tensor_info = self.analyze_tensors_quick()
                    
                    # Update history
                    self.vram_history.append(gpu_stats['mem_used'])
                    
                    # Update display
                    layout = self.create_layout(gpu_stats, proc_info, tensor_info)
                    live.update(layout)
                    
                    time.sleep(self.update_interval)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
                    time.sleep(self.update_interval)
        
        console.print("\n[yellow]Monitoring stopped[/yellow]")


if __name__ == "__main__":
    try:
        # Parse update interval from command line
        update_interval = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
        
        monitor = GPUMonitor(update_interval=update_interval)
        monitor.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting...[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]Fatal Error: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)
