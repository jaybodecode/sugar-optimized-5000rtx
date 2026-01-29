#!/usr/bin/env python3
"""
Enhanced System Monitor - Compact Rich UI
Monitors: VRAM, CPU, RAM, Disk I/O, Network I/O, and training process
Usage: python monitor_system.py [--logfile path/to/log.csv] [--interval 5]
"""

import argparse
import csv
import os
import psutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text


class SystemMonitor:
    def __init__(self, interval: int = 5, logfile: Optional[str] = None):
        self.interval = interval
        self.logfile = logfile
        self.console = Console()
        self.training_pid: Optional[int] = None
        self.prev_vmsize_mb: int = 0
        self.prev_net_io: Optional[Dict] = None
        self.prev_disk_io: Optional[Dict] = None
        self.start_time = time.time()
        
        if self.logfile:
            self._init_logfile()
    
    def _init_logfile(self):
        """Initialize CSV log file with headers"""
        with open(self.logfile, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestamp', 'VRAM_GB', 'GPU_Util%', 'Mem_Util%', 'PCIe_Gen', 'PCIe_Width',
                'CPU%', 'Process_RSS_GB', 'Process_VmSize_GB', 'VmSize_Delta_MB',
                'System_RAM_GB', 'RAM_Used%', 'Disk_Read_MB/s', 'Disk_Write_MB/s',
                'Net_Recv_MB/s', 'Net_Send_MB/s'
            ])
    
    def find_training_process(self) -> Optional[int]:
        """Find the train.py process PID"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and 'train.py' in ' '.join(cmdline) and 'conda' not in ' '.join(cmdline):
                    return proc.info['pid']
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None
    
    def get_gpu_stats(self) -> Dict:
        """Get GPU memory, utilization, and PCIe stats via nvidia-smi"""
        try:
            # Query: memory.used, utilization.gpu, utilization.memory
            result = subprocess.run(
                ['nvidia-smi', 
                 '--query-gpu=memory.used,utilization.gpu,utilization.memory', 
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = [p.strip() for p in result.stdout.strip().split(',')]
                vram_mb = float(parts[0]) if parts[0] else 0
                gpu_util = int(parts[1]) if parts[1] else 0
                mem_util = int(parts[2]) if parts[2] else 0
                
                # Try to get PCIe throughput (KB/s)
                pcie_tx_kb = 0
                pcie_rx_kb = 0
                try:
                    pcie_result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=pcie.link.gen.current,pcie.link.width.current',
                         '--format=csv,noheader,nounits'],
                        capture_output=True, text=True, timeout=2
                    )
                    if pcie_result.returncode == 0:
                        pcie_parts = [p.strip() for p in pcie_result.stdout.strip().split(',')]
                        pcie_gen = int(pcie_parts[0]) if pcie_parts[0] else 0
                        pcie_width = int(pcie_parts[1]) if pcie_parts[1] else 0
                    else:
                        pcie_gen = 0
                        pcie_width = 0
                except:
                    pcie_gen = 0
                    pcie_width = 0
                
                return {
                    'vram_gb': vram_mb / 1024,
                    'gpu_util': gpu_util,
                    'mem_util': mem_util,
                    'pcie_gen': pcie_gen,
                    'pcie_width': pcie_width
                }
        except Exception as e:
            pass
        return {
            'vram_gb': 0.0, 
            'gpu_util': 0, 
            'mem_util': 0,
            'pcie_gen': 0,
            'pcie_width': 0
        }
    
    def get_process_memory(self, pid: int) -> Dict:
        """Get process memory stats from /proc"""
        try:
            with open(f'/proc/{pid}/status', 'r') as f:
                status = f.read()
            
            vmrss_kb = 0
            vmsize_kb = 0
            
            for line in status.split('\n'):
                if line.startswith('VmRSS:'):
                    vmrss_kb = int(line.split()[1])
                elif line.startswith('VmSize:'):
                    vmsize_kb = int(line.split()[1])
            
            proc_rss_mb = vmrss_kb // 1024
            proc_vmsize_mb = vmsize_kb // 1024
            
            delta_mb = 0
            if self.prev_vmsize_mb > 0:
                delta_mb = proc_vmsize_mb - self.prev_vmsize_mb
            self.prev_vmsize_mb = proc_vmsize_mb
            
            return {
                'rss_gb': proc_rss_mb / 1024,
                'vmsize_gb': proc_vmsize_mb / 1024,
                'delta_mb': delta_mb
            }
        except Exception:
            return None
    
    def get_disk_io(self) -> Dict:
        """Get disk I/O stats for all disks"""
        disk_io = psutil.disk_io_counters(perdisk=False)
        
        if disk_io and self.prev_disk_io:
            time_delta = self.interval
            read_mb = (disk_io.read_bytes - self.prev_disk_io.read_bytes) / (1024 * 1024) / time_delta
            write_mb = (disk_io.write_bytes - self.prev_disk_io.write_bytes) / (1024 * 1024) / time_delta
            
            self.prev_disk_io = disk_io
            return {'read_mb_s': read_mb, 'write_mb_s': write_mb}
        
        self.prev_disk_io = disk_io
        return {'read_mb_s': 0.0, 'write_mb_s': 0.0}
    
    def get_network_io(self) -> Dict:
        """Get network I/O stats for all interfaces"""
        net_io = psutil.net_io_counters()
        
        if net_io and self.prev_net_io:
            time_delta = self.interval
            recv_mb = (net_io.bytes_recv - self.prev_net_io.bytes_recv) / (1024 * 1024) / time_delta
            sent_mb = (net_io.bytes_sent - self.prev_net_io.bytes_sent) / (1024 * 1024) / time_delta
            
            self.prev_net_io = net_io
            return {'recv_mb_s': recv_mb, 'sent_mb_s': sent_mb}
        
        self.prev_net_io = net_io
        return {'recv_mb_s': 0.0, 'sent_mb_s': 0.0}
    
    def create_gpu_table(self, data: Dict) -> Table:
        """Create GPU-specific stats table"""
        table = Table(
            title="GPU Stats",
            show_header=True,
            header_style="bold green",
            border_style="bright_green",
            title_style="bold green"
        )
        
        table.add_column("Status", style="green", width=8)
        table.add_column("VRAM", justify="right", width=9)
        table.add_column("GPU%", justify="right", width=6)
        table.add_column("MemBus%", justify="right", width=8)
        table.add_column("PCIe", justify="right", width=9)
        
        gpu_stats = data.get('gpu', {})
        
        # GPU icon
        gpu_icon = "🟢 Active" if gpu_stats.get('gpu_util', 0) > 0 else "⚪ Idle"
        
        # PCIe info
        pcie_gen = gpu_stats.get('pcie_gen', 0)
        pcie_width = gpu_stats.get('pcie_width', 0)
        pcie_str = f"G{pcie_gen}x{pcie_width}" if pcie_gen > 0 and pcie_width > 0 else "N/A"
        
        table.add_row(
            gpu_icon,
            f"{gpu_stats.get('vram_gb', 0):.2f} GB",
            f"{gpu_stats.get('gpu_util', 0)}%",
            f"{gpu_stats.get('mem_util', 0)}%",
            pcie_str
        )
        
        return table
    
    def create_system_table(self, data: Dict) -> Table:
        """Create system and process stats table"""
        table = Table(
            title=f"System Monitor (Interval: {self.interval}s) - Uptime: {int(time.time() - self.start_time)}s",
            show_header=True,
            header_style="bold cyan",
            border_style="bright_blue",
            title_style="bold magenta"
        )
        
        # CPU/RAM Section
        table.add_column("CPU%", justify="right", width=6)
        table.add_column("RAM", justify="right", width=9)
        table.add_column("RAM%", justify="right", width=6)
        
        # Process Section
        table.add_column("P.RSS", justify="right", width=8)
        table.add_column("P.VmSz", justify="right", width=8)
        table.add_column("Δ MB", justify="right", width=8)
        
        # I/O Section
        table.add_column("Disk R", justify="right", width=7)
        table.add_column("Disk W", justify="right", width=7)
        table.add_column("Net R", justify="right", width=7)
        table.add_column("Net S", justify="right", width=7)
        
        # Format data
        proc_mem = data.get('process', {})
        ram = data.get('ram', {})
        disk = data.get('disk', {})
        net = data.get('network', {})
        
        # Format delta with color
        delta_mb = proc_mem.get('delta_mb', 0) if proc_mem else 0
        if delta_mb > 0:
            delta_str = f"[green]+{delta_mb}[/green]"
        elif delta_mb < 0:
            delta_str = f"[red]{delta_mb}[/red]"
        else:
            delta_str = "0"
        
        table.add_row(
            f"{data.get('cpu', 0):.1f}%",
            f"{ram.get('used_gb', 0):.1f} GB",
            f"{ram.get('percent', 0):.1f}%",
            f"{proc_mem.get('rss_gb', 0):.2f} GB" if proc_mem else "N/A",
            f"{proc_mem.get('vmsize_gb', 0):.2f} GB" if proc_mem else "N/A",
            delta_str if proc_mem else "N/A",
            f"{disk.get('read_mb_s', 0):.1f}",
            f"{disk.get('write_mb_s', 0):.1f}",
            f"{net.get('recv_mb_s', 0):.1f}",
            f"{net.get('sent_mb_s', 0):.1f}"
        )
        
        return table
    
    def collect_stats(self) -> Dict:
        """Collect all system statistics"""
        # Find training process
        if not self.training_pid or not psutil.pid_exists(self.training_pid):
            self.training_pid = self.find_training_process()
        
        # Collect stats
        gpu_stats = self.get_gpu_stats()
        cpu_percent = psutil.cpu_percent(interval=None)
        
        ram = psutil.virtual_memory()
        ram_stats = {
            'used_gb': ram.used / (1024 ** 3),
            'percent': ram.percent
        }
        
        proc_mem = None
        if self.training_pid:
            proc_mem = self.get_process_memory(self.training_pid)
        
        disk_io = self.get_disk_io()
        net_io = self.get_network_io()
        
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'gpu': gpu_stats,
            'cpu': cpu_percent,
            'ram': ram_stats,
            'process': proc_mem,
            'disk': disk_io,
            'network': net_io
        }
    
    def log_stats(self, stats: Dict):
        """Log statistics to CSV file"""
        if not self.logfile:
            return
        
        proc = stats.get('process')
        gpu = stats['gpu']
        with open(self.logfile, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                stats['timestamp'],
                f"{gpu['vram_gb']:.2f}",
                gpu['gpu_util'],
                gpu['mem_util'],
                gpu['pcie_gen'],
                gpu['pcie_width'],
                f"{stats['cpu']:.1f}",
                f"{proc['rss_gb']:.2f}" if proc else 'N/A',
                f"{proc['vmsize_gb']:.2f}" if proc else 'N/A',
                proc['delta_mb'] if proc else 0,
                f"{stats['ram']['used_gb']:.1f}",
                f"{stats['ram']['percent']:.1f}",
                f"{stats['disk']['read_mb_s']:.2f}",
                f"{stats['disk']['write_mb_s']:.2f}",
                f"{stats['network']['recv_mb_s']:.2f}",
                f"{stats['network']['sent_mb_s']:.2f}"
            ])
    
    def create_info_panel(self) -> Panel:
        """Create info panel with legend and status"""
        info_text = Text()
        info_text.append("🟢 GPU Active  ", style="green")
        info_text.append("⚪ GPU Idle  ", style="white")
        info_text.append(f"PID: {self.training_pid or 'Searching...'}", style="yellow")
        if self.logfile:
            info_text.append(f"  Log: {Path(self.logfile).name}", style="cyan")
        
        return Panel(info_text, title="Status", border_style="blue")
    
    def run(self):
        """Main monitoring loop with Rich Live display"""
        self.console.print("\n[bold cyan]System Monitor Starting...[/bold cyan]")
        self.console.print(f"Interval: {self.interval}s | Press Ctrl+C to stop\n")
        
        # Initialize counters for first reading
        self.get_disk_io()
        self.get_network_io()
        time.sleep(1)  # Initial sampling period
        
        try:
            with Live(console=self.console, refresh_per_second=1) as live:
                while True:
                    stats = self.collect_stats()
                    
                    # Create layout with two tables
                    layout = Layout()
                    layout.split_column(
                        Layout(self.create_info_panel(), size=3),
                        Layout(self.create_gpu_table(stats), size=6),
                        Layout(self.create_system_table(stats))
                    )
                    
                    live.update(layout)
                    self.log_stats(stats)
                    
                    time.sleep(self.interval)
        
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Monitoring stopped by user[/yellow]")


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced System Monitor with Rich UI'
    )
    parser.add_argument(
        '--logfile',
        type=str,
        help='Path to CSV log file for recording stats'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=5,
        help='Sampling interval in seconds (default: 5)'
    )
    
    args = parser.parse_args()
    
    monitor = SystemMonitor(interval=args.interval, logfile=args.logfile)
    monitor.run()


if __name__ == '__main__':
    main()
