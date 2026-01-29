#!/usr/bin/env python3
"""
Simple GPU Memory Monitor - Prints stats every N seconds
"""

import torch
import psutil
import time
import sys
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

def find_training_process():
    """Find active training process"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
        try:
            cmdline = proc.info.get('cmdline', [])
            if cmdline and any('train.py' in str(arg) for arg in cmdline):
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None

def print_stats():
    """Print current GPU and process stats"""
    console.clear()
    console.print(f"[bold cyan]GPU Monitor[/bold cyan] - {datetime.now().strftime('%H:%M:%S')}\n")
    
    # GPU Stats
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        mem_free, mem_total = torch.cuda.mem_get_info()
        mem_used = (mem_total - mem_free) / (1024**3)
        mem_total_gb = mem_total / (1024**3)
        util = (mem_used / mem_total_gb) * 100
        
        table = Table(title="GPU Memory", box=box.SIMPLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total", f"{mem_total_gb:.2f} GB")
        table.add_row("Used", f"{mem_used:.2f} GB ({util:.1f}%)")
        table.add_row("Allocated", f"{allocated:.2f} GB")
        table.add_row("Reserved", f"{reserved:.2f} GB")
        table.add_row("Fragmentation", f"{reserved - allocated:.2f} GB")
        
        console.print(table)
    
    # Process Stats
    proc = find_training_process()
    if proc:
        try:
            cpu = proc.cpu_percent()
            ram_gb = proc.memory_info().rss / (1024**3)
            
            proc_table = Table(title="Training Process", box=box.SIMPLE)
            proc_table.add_column("Metric", style="cyan")
            proc_table.add_column("Value", style="yellow")
            
            proc_table.add_row("PID", str(proc.pid))
            proc_table.add_row("CPU", f"{cpu:.1f}%")
            proc_table.add_row("RAM", f"{ram_gb:.2f} GB")
            
            console.print("\n")
            console.print(proc_table)
            
            # Command line
            cmdline = ' '.join(proc.cmdline())
            if len(cmdline) > 120:
                cmdline = cmdline[:117] + "..."
            console.print(f"\n[dim]{cmdline}[/dim]")
        except:
            console.print("\n[yellow]Process info unavailable[/yellow]")
    else:
        console.print("\n[red]No training process found[/red]")
    
    console.print("\n[dim]Press Ctrl+C to exit[/dim]")

def main():
    interval = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    
    console.print("[green]Starting GPU monitor...[/green]\n")
    time.sleep(1)
    
    try:
        while True:
            print_stats()
            time.sleep(interval)
    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting...[/yellow]")

if __name__ == "__main__":
    main()
