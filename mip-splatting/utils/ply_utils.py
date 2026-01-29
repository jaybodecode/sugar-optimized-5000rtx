"""
PLY Analysis and Cleanup Utilities

Functions for analyzing and cleaning Gaussian Splatting PLY files
integrated into the training pipeline.
"""

import os
import sys
import subprocess


def analyze_ply_with_metrics(ply_path, iteration=None, loss=None, psnr=None, ssim=None, lpips=None, output_file=None):
    """
    Run PLY analysis with training metrics.
    
    Args:
        ply_path: Path to PLY file
        iteration: Training iteration number
        loss: Training loss value
        psnr: PSNR metric
        ssim: SSIM metric
        lpips: LPIPS metric
        output_file: Path to write analysis output (if None, writes to stdout)
    
    Returns:
        True if successful, False otherwise
    """
    analyze_cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "TOOLS", "analyze_ply.py"),
        str(ply_path),
    ]
    
    if iteration is not None:
        analyze_cmd.extend(["--iteration", str(iteration)])
    if loss is not None:
        analyze_cmd.extend(["--loss", f"{loss:.6f}"])
    if psnr is not None:
        analyze_cmd.extend(["--psnr", f"{psnr:.2f}"])
    if ssim is not None:
        analyze_cmd.extend(["--ssim", f"{ssim:.4f}"])
    if lpips is not None:
        analyze_cmd.extend(["--lpips", f"{lpips:.4f}"])
    
    try:
        if output_file:
            with open(output_file, 'w') as f:
                subprocess.run(analyze_cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
        else:
            subprocess.run(analyze_cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  PLY analysis failed: {e}")
        return False
    except Exception as e:
        print(f"⚠️  PLY analysis error: {e}")
        return False


def cleanup_ply(input_path, output_path, min_opacity=-5.0, max_scale=10.0, bbox_expansion=1.5, output_file=None):
    """
    Clean up a PLY file by removing low-opacity, huge-scale, and outlier Gaussians.
    
    Args:
        input_path: Path to input PLY
        output_path: Path to output cleaned PLY
        min_opacity: Minimum opacity threshold (logit space)
        max_scale: Maximum scale magnitude threshold
        bbox_expansion: Bounding box expansion factor
        output_file: Path to write cleanup output (if None, writes to stdout)
    
    Returns:
        True if successful, False otherwise
    """
    cleanup_cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "TOOLS", "cleanup_ply.py"),
        str(input_path),
        str(output_path),
        "--min-opacity", str(min_opacity),
        "--max-scale", str(max_scale),
        "--bbox-expansion", str(bbox_expansion)
    ]
    
    try:
        if output_file:
            with open(output_file, 'w') as f:  # Write mode (separate file for cleanup)
                subprocess.run(cleanup_cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
        else:
            subprocess.run(cleanup_cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  PLY cleanup failed: {e}")
        return False
    except Exception as e:
        print(f"⚠️  PLY cleanup error: {e}")
        return False


def process_saved_ply(ply_path, iteration, loss, test_metrics=None, console=None):
    """
    Complete workflow: Analyze original PLY, cleanup, save cleaned, analyze cleaned.
    Writes analysis output to separate files:
      - 'ply_analysis_original.txt' for original PLY
      - 'ply_analysis_cleaned.txt' for cleaned PLY
    
    Args:
        ply_path: Path to saved PLY file
        iteration: Training iteration number
        loss: Training loss value
        test_metrics: Dict with 'psnr', 'ssim', 'lpips' keys (optional)
        console: Rich console for progress messages (optional)
    
    Returns:
        Dict with success status and cleaned PLY path
    """
    if not os.path.exists(ply_path):
        if console:
            console.print("[yellow]⚠️  PLY file not found: {}[/yellow]".format(ply_path))
        return {'success': False, 'cleaned_path': None}
    
    test_metrics = test_metrics or {}
    ply_dir = os.path.dirname(ply_path)
    original_analysis = os.path.join(ply_dir, "ply_analysis_original.txt")
    cleaned_analysis = os.path.join(ply_dir, "ply_analysis_cleaned.txt")
    cleanup_log = os.path.join(ply_dir, "ply_cleanup.txt")
    
    # Simple console message
    if console:
        console.print(f"[cyan]🔍 Analyzing and cleaning PLY (output: ply_analysis_*.txt)[/cyan]")
    
    # 1. Analyze original (write to original file)
    analyze_ply_with_metrics(
        ply_path,
        iteration=iteration,
        loss=loss,
        psnr=test_metrics.get('psnr'),
        ssim=test_metrics.get('ssim'),
        lpips=test_metrics.get('lpips'),
        output_file=original_analysis
    )
    
    # 2. Cleanup and save (write to cleanup log)
    cleaned_ply_path = os.path.join(ply_dir, "point_cloud_cleaned.ply")
    
    success = cleanup_ply(
        ply_path,
        cleaned_ply_path,
        min_opacity=-5.0,  # Same as training: sigmoid(-5) ≈ 0.0067
        max_scale=10.0,
        bbox_expansion=1.5,
        output_file=cleanup_log
    )
    
    if not success:
        return {'success': False, 'cleaned_path': None}
    
    # 3. Analyze cleaned (write to cleaned file)
    analyze_ply_with_metrics(
        cleaned_ply_path,
        iteration=iteration,
        loss=loss,
        psnr=test_metrics.get('psnr'),
        ssim=test_metrics.get('ssim'),
        lpips=test_metrics.get('lpips'),
        output_file=cleaned_analysis
    )
    
    if console:
        console.print(f"[green]✓ PLY analysis complete: {ply_dir}/*.txt[/green]")
    
    return {'success': True, 'cleaned_path': cleaned_ply_path}
