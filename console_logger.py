"""
Shared console logger for SugarV3 projects (mip-splatting and SuGaR)

Provides timestamped console output with Rich formatting support.
Prevents duplicate timestamps from IPython/Jupyter environments.
"""

import sys
from datetime import datetime
from rich.console import Console

# Create global console instance
_CONSOLE = Console(width=120, force_terminal=True, force_interactive=True, legacy_windows=False)

def log(message="", style=""):
    """Print message with timestamp prefix in dim color.
    
    Args:
        message: Text to print (can include Rich markup)
        style: Additional Rich style to apply to entire line
    """
    if message:
        timestamp = datetime.now().strftime("%d/%m %H:%M:%S")
        # Use grey color (RGB 128,128,128) for timestamp consistency across terminals
        _CONSOLE.print(f"[rgb(128,128,128)]{timestamp}[/rgb(128,128,128)] {message}", highlight=False, markup=True)
    else:
        _CONSOLE.print()

def print_table(table):
    """Print Rich table without timestamp (tables span multiple lines)."""
    _CONSOLE.print(table)

def print_progress(text):
    """Print progress text with timestamp, updates same line (carriage return)."""
    timestamp = datetime.now().strftime("%d/%m %H:%M:%S")
    # Use ANSI escape codes for grey color (consistent with Rich output)
    sys.stdout.write(f"\r\033[38;2;128;128;128m{timestamp}\033[0m {text}")
    sys.stdout.flush()

def print_progress_end():
    """Finish progress line with newline."""
    sys.stdout.write("\n")
    sys.stdout.flush()

# Singleton pattern - expose console for advanced usage (Progress bars, etc)
def get_console():
    """Get the Rich Console instance for advanced features like Progress bars."""
    return _CONSOLE

# Convenience shortcuts
def blank():
    """Print a blank line."""
    log()

def info(msg):
    """Print info message."""
    log(f"[cyan]{msg}[/cyan]")

def success(msg):
    """Print success message."""
    log(f"[green]{msg}[/green]")

def warning(msg):
    """Print warning message."""
    log(f"[yellow]{msg}[/yellow]")

def error(msg):
    """Print error message."""
    log(f"[red bold]{msg}[/red bold]")
