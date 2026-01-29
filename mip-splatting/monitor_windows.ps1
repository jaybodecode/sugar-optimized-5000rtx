# Windows PowerShell Monitor for WSL2 Training
# Run this in PowerShell on Windows (not WSL2)
# Usage: .\monitor_windows.ps1 [output.csv]

param(
    [string]$LogFile = ""
)

$Interval = 5
$PrevWSLMemMB = 0

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Windows System Monitor - Sampling every ${Interval}s" -ForegroundColor Cyan
Write-Host "Tracking: GPU VRAM, WSL2 Memory (real RAM including CUDA)" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if nvidia-smi is available
try {
    nvidia-smi --version | Out-Null
    $HasNVIDIA = $true
} catch {
    Write-Host "⚠ Warning: nvidia-smi not found in PATH" -ForegroundColor Yellow
    $HasNVIDIA = $false
}

# Print header
Write-Host ("{0,-19} | {1,8} | {2,5} | {3,9} | {4,6} | {5,8}" -f "Timestamp", "VRAM", "GPU%", "WSL2 RAM", "Δ RAM", "Total RAM")
Write-Host ("{0,-19}-+-{1,8}-+-{2,5}-+-{3,9}-+-{4,6}-+-{5,8}" -f "-"*19, "-"*8, "-"*5, "-"*9, "-"*6, "-"*8)

# Write CSV header if logging
if ($LogFile -ne "") {
    "Timestamp,VRAM_GB,GPU_Util%,WSL2_RAM_GB,RAM_Delta_MB,Total_RAM_GB" | Out-File -FilePath $LogFile -Encoding UTF8
}

while ($true) {
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    
    # Get GPU stats
    if ($HasNVIDIA) {
        try {
            $GPUInfo = nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits
            $GPUParts = $GPUInfo -split ','
            $VRAM_MB = [int]$GPUParts[0].Trim()
            $VRAM_GB = [math]::Round($VRAM_MB / 1024, 2)
            $GPU_Util = $GPUParts[1].Trim()
        } catch {
            $VRAM_GB = 0
            $GPU_Util = 0
        }
    } else {
        $VRAM_GB = 0
        $GPU_Util = 0
    }
    
    # Get WSL2 memory (vmmem process + python processes)
    $WSLProcs = Get-Process | Where-Object { 
        $_.ProcessName -eq "vmmem" -or 
        ($_.ProcessName -eq "python" -and $_.Path -like "*WSL*")
    }
    
    if ($WSLProcs) {
        $WSLMemBytes = ($WSLProcs | Measure-Object WorkingSet64 -Sum).Sum
        $WSLMemMB = [math]::Round($WSLMemBytes / 1MB, 0)
        $WSLMemGB = [math]::Round($WSLMemMB / 1024, 2)
        
        # Calculate delta
        if ($PrevWSLMemMB -gt 0) {
            $DeltaMB = $WSLMemMB - $PrevWSLMemMB
        } else {
            $DeltaMB = 0
        }
        $PrevWSLMemMB = $WSLMemMB
    } else {
        $WSLMemGB = 0
        $DeltaMB = 0
    }
    
    # Get total system RAM
    $TotalRAM = Get-CimInstance Win32_OperatingSystem
    $TotalRAMGB = [math]::Round($TotalRAM.TotalVisibleMemorySize / 1MB, 1)
    $UsedRAMGB = [math]::Round(($TotalRAM.TotalVisibleMemorySize - $TotalRAM.FreePhysicalMemory) / 1MB, 1)
    
    # Format delta
    if ($DeltaMB -gt 0) {
        $DeltaStr = "+{0,5}" -f $DeltaMB
    } elseif ($DeltaMB -lt 0) {
        $DeltaStr = "{0,6}" -f $DeltaMB
    } else {
        $DeltaStr = "     0"
    }
    
    # Print to console
    $Line = "{0,-19} | {1,6:F2} GB | {2,4}% | {3,7:F2} GB | {4,6} | {5,6:F1} GB" -f `
        $Timestamp, $VRAM_GB, $GPU_Util, $WSLMemGB, $DeltaStr, $UsedRAMGB
    Write-Host $Line
    
    # Log to CSV if specified
    if ($LogFile -ne "" -and $WSLMemGB -gt 0) {
        "$Timestamp,$VRAM_GB,$GPU_Util,$WSLMemGB,$DeltaMB,$UsedRAMGB" | Out-File -FilePath $LogFile -Append -Encoding UTF8
    }
    
    Start-Sleep -Seconds $Interval
}
