#!/usr/bin/env python3
"""
System Specifications Display for SuGaR v3 Training Environment
Displays hardware and software specs used for benchmarking
"""

import subprocess
import sys
import platform
import psutil
import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


def get_cpu_info():
    """Get CPU information"""
    try:
        # Try to get CPU model from /proc/cpuinfo
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if 'model name' in line:
                    cpu_model = line.split(':')[1].strip()
                    break
        
        # Get core count
        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)
        
        return {
            'model': cpu_model,
            'physical_cores': physical_cores,
            'logical_cores': logical_cores,
            'max_freq': f"{psutil.cpu_freq().max:.0f} MHz" if psutil.cpu_freq() else "N/A"
        }
    except Exception as e:
        return {'model': 'Unknown', 'physical_cores': 'N/A', 'logical_cores': 'N/A', 'max_freq': 'N/A'}


def get_ram_info():
    """Get RAM information"""
    mem = psutil.virtual_memory()
    return {
        'total': f"{mem.total / (1024**3):.1f} GB",
        'available': f"{mem.available / (1024**3):.1f} GB",
        'speed': get_ram_speed()
    }


def get_ram_speed():
    """Try to get RAM speed from dmidecode"""
    try:
        result = subprocess.run(['sudo', 'dmidecode', '-t', 'memory'], 
                              capture_output=True, text=True, timeout=5)
        for line in result.stdout.split('\n'):
            if 'Speed:' in line and 'MHz' in line:
                return line.split(':')[1].strip()
        return "Unknown"
    except:
        return "Unknown (run with sudo for RAM speed)"


def get_disk_info():
    """Get disk information"""
    try:
        # Get root partition info
        disk = psutil.disk_usage('/')
        
        # Try to get NVMe info
        nvme_info = get_nvme_info()
        
        return {
            'total': f"{disk.total / (1024**3):.1f} GB",
            'used': f"{disk.used / (1024**3):.1f} GB",
            'free': f"{disk.free / (1024**3):.1f} GB",
            'type': nvme_info
        }
    except Exception as e:
        return {'total': 'N/A', 'used': 'N/A', 'free': 'N/A', 'type': 'Unknown'}


def get_nvme_info():
    """Try to detect NVMe drives"""
    try:
        result = subprocess.run(['lsblk', '-d', '-o', 'NAME,ROTA'], 
                              capture_output=True, text=True, timeout=5)
        for line in result.stdout.split('\n'):
            if 'nvme' in line.lower() and '0' in line:
                return "NVMe SSD"
        return "SSD/HDD"
    except:
        return "Unknown"


def get_pcie_info():
    """Get PCIe information for GPU"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=pcie.link.gen.current,pcie.link.width.current', 
                               '--format=csv,noheader'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gen, width = result.stdout.strip().split(', ')
            return f"PCIe Gen{gen} x{width}"
        return "Unknown"
    except:
        return "Unknown"


def get_gpu_info():
    """Get GPU information"""
    try:
        # Get GPU name
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', 
                               '--format=csv,noheader'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            name, mem, driver = result.stdout.strip().split(', ')
            
            # Get memory clock
            mem_clock = subprocess.run(['nvidia-smi', '--query-gpu=clocks.mem', 
                                       '--format=csv,noheader'], 
                                      capture_output=True, text=True, timeout=5)
            mem_clock_speed = mem_clock.stdout.strip() if mem_clock.returncode == 0 else "Unknown"
            
            # Get CUDA version
            cuda_version = torch.version.cuda if torch.cuda.is_available() else "Not available"
            
            # Get compute capability
            if torch.cuda.is_available():
                compute_cap = torch.cuda.get_device_capability(0)
                arch = f"sm_{compute_cap[0]}{compute_cap[1]}"
            else:
                arch = "N/A"
            
            return {
                'name': name,
                'memory': mem,
                'driver': driver,
                'mem_clock': mem_clock_speed,
                'cuda': cuda_version,
                'arch': arch,
                'pcie': get_pcie_info()
            }
        return None
    except Exception as e:
        return None


def get_python_env():
    """Get Python environment information"""
    import torch
    
    try:
        import pytorch3d
        pytorch3d_version = pytorch3d.__version__
    except:
        pytorch3d_version = "Not installed"
    
    try:
        import nvdiffrast
        nvdiffrast_version = nvdiffrast.__version__ if hasattr(nvdiffrast, '__version__') else "Installed"
    except:
        nvdiffrast_version = "Not installed"
    
    return {
        'python': platform.python_version(),
        'pytorch': torch.__version__,
        'pytorch3d': pytorch3d_version,
        'nvdiffrast': nvdiffrast_version,
        'cuda_available': torch.cuda.is_available()
    }


def display_system_specs():
    """Display all system specifications in a nice format"""
    from rich.columns import Columns
    from rich.console import Group
    
    console.print(Panel.fit(
        "[bold cyan]🖥️  SuGaR v3 Training Environment - System Specifications[/bold cyan]",
        border_style="bright_cyan"
    ))
    
    # Get all info first
    cpu_info = get_cpu_info()
    ram_info = get_ram_info()
    disk_info = get_disk_info()
    gpu_info = get_gpu_info()
    env_info = get_python_env()
    
    # CPU Information
    cpu_table = Table(title="🔧 CPU", box=box.HEAVY, border_style="bright_green", show_header=False, padding=0)
    cpu_table.add_column("Property", style="cyan", no_wrap=True)
    cpu_table.add_column("Value", style="bright_yellow")
    cpu_table.add_row("Model", cpu_info['model'])
    cpu_table.add_row("Cores", f"{cpu_info['physical_cores']}P / {cpu_info['logical_cores']}L")
    
    # RAM Information
    ram_table = Table(title="💾 RAM", box=box.HEAVY, border_style="bright_blue", show_header=False, padding=0)
    ram_table.add_column("Property", style="cyan", no_wrap=True)
    ram_table.add_column("Value", style="bright_yellow")
    ram_table.add_row("Total", ram_info['total'])
    ram_table.add_row("Available", ram_info['available'])
    
    # Disk Information
    disk_table = Table(title="💿 Storage", box=box.HEAVY, border_style="bright_magenta", show_header=False, padding=0)
    disk_table.add_column("Property", style="cyan", no_wrap=True)
    disk_table.add_column("Value", style="bright_yellow")
    disk_table.add_row("Capacity", disk_info['total'])
    disk_table.add_row("Free", disk_info['free'])
    disk_table.add_row("Type", disk_info['type'])
    
    # Print first row: CPU, RAM, Disk side by side
    console.print(Columns([cpu_table, ram_table, disk_table], equal=True, expand=True))
    
    # GPU Information
    if gpu_info:
        gpu_table = Table(title="🎮 GPU", box=box.HEAVY, border_style="bright_red", show_header=False, padding=0)
        gpu_table.add_column("Property", style="cyan", no_wrap=True)
        gpu_table.add_column("Value", style="bright_yellow")
        gpu_table.add_row("Model", gpu_info['name'])
        gpu_table.add_row("VRAM", gpu_info['memory'])
        gpu_table.add_row("Driver", gpu_info['driver'])
        gpu_table.add_row("Mem Clock", gpu_info['mem_clock'])
        gpu_table.add_row("CUDA", gpu_info['cuda'])
        gpu_table.add_row("Arch", gpu_info['arch'])
        gpu_table.add_row("PCIe", gpu_info['pcie'])
    else:
        gpu_table = Panel("[red]⚠️  No GPU detected[/red]", border_style="red")
    
    # Python Environment
    env_table = Table(title="🐍 Python", box=box.HEAVY, border_style="bright_yellow", show_header=False, padding=0)
    env_table.add_column("Package", style="cyan", no_wrap=True)
    env_table.add_column("Version", style="bright_yellow")
    env_table.add_row("Python", env_info['python'])
    env_table.add_row("PyTorch", env_info['pytorch'])
    env_table.add_row("PyTorch3D", env_info['pytorch3d'])
    env_table.add_row("nvdiffrast", env_info['nvdiffrast'])
    env_table.add_row("CUDA", "✅ Yes" if env_info['cuda_available'] else "❌ No")
    
    # Print second row: GPU and Python side by side
    console.print(Columns([gpu_table, env_table], equal=True, expand=True))
    
    # Summary
    console.print(Panel.fit(
        f"[bold bright_green]✅ System Ready for SuGaR v3 Training[/bold bright_green] - [bright_cyan]{gpu_info['name'] if gpu_info else 'Unknown GPU'}[/bright_cyan]",
        border_style="bright_green"
    ))


if __name__ == "__main__":
    try:
        display_system_specs()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)
