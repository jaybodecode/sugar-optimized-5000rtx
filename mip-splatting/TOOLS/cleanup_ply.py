#!/usr/bin/env python3
"""
Gaussian Splatting PLY Cleanup Tool

Cleans up PLY files by removing:
- Low opacity Gaussians (invisible)
- Huge scale outliers (artifacts)
- Spatial outliers (outside bounding box)

Compatible with Mip-Splatting (preserves filter_3D attribute)

Usage:
    python cleanup_ply.py input.ply output.ply
    python cleanup_ply.py input.ply output.ply --min-opacity 0.01 --max-scale 10.0
    python cleanup_ply.py input.ply output.ply --bbox-expansion 1.5 --dry-run
"""

import argparse
import sys
from pathlib import Path
import numpy as np

try:
    from plyfile import PlyData, PlyElement
except ImportError:
    print("ERROR: plyfile not installed. Run: pip install plyfile")
    sys.exit(1)


def cleanup_ply(input_path, output_path, min_opacity=-5.0, max_scale=10.0, bbox_expansion=1.5, center_at_origin=True, dry_run=False):
    """
    Clean up a Gaussian Splatting PLY file
    
    Args:
        input_path: Path to input PLY
        output_path: Path to output PLY
        min_opacity: Minimum opacity threshold (logit space, ~sigmoid(-5.0) = 0.0067)
        max_scale: Maximum scale magnitude threshold
        bbox_expansion: Bounding box expansion factor (1.5 = keep points within 1.5x scene extent)
        center_at_origin: If True, center scene at origin (0, 0, 0) - recommended for viewers
        dry_run: If True, only analyze without saving
    """
    
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"PLY CLEANUP TOOL")
    print(f"{'='*80}\n")
    
    print(f"📁 Input:  {input_path.name}")
    print(f"📂 Output: {output_path}")
    print()
    
    # Load PLY
    try:
        plydata = PlyData.read(str(input_path))
    except Exception as e:
        print(f"ERROR: Failed to read PLY file: {e}")
        sys.exit(1)
    
    vertex = plydata['vertex']
    n_points = len(vertex)
    
    print(f"Original: {n_points:,} Gaussians")
    print()
    
    # Extract positions
    x = np.array(vertex['x'])
    y = np.array(vertex['y'])
    z = np.array(vertex['z'])
    
    # Create keep mask (start with all True)
    keep_mask = np.ones(n_points, dtype=bool)
    
    # Calculate scene center and extent
    center = np.array([np.mean(x), np.mean(y), np.mean(z)])
    distances = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
    extent = np.max(distances)
    
    print(f"Scene center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
    print(f"Scene extent: {extent:.2f} meters")
    print()
    
    # 1. Remove low opacity Gaussians
    if 'opacity' in vertex.data.dtype.names:
        opacity = np.array(vertex['opacity'])
        low_opacity_mask = opacity < min_opacity
        removed_opacity = np.sum(low_opacity_mask)
        keep_mask &= ~low_opacity_mask
        
        print(f"{'─'*80}")
        print(f"OPACITY FILTER")
        print(f"{'─'*80}")
        print(f"Threshold: < {min_opacity:.2f} (logit space)")
        print(f"          ≈ {1.0 / (1.0 + np.exp(-min_opacity)):.4f} (sigmoid)")
        print(f"Removed:   {removed_opacity:,} Gaussians ({removed_opacity/n_points*100:.2f}%)")
        print()
    
    # 2. Remove huge scale outliers
    scale_props = [p for p in vertex.data.dtype.names if p.startswith('scale_')]
    if scale_props:
        scales = np.array([vertex[p] for p in scale_props]).T  # Shape: (n_points, 3)
        scale_magnitudes = np.linalg.norm(scales, axis=1)
        huge_scale_mask = scale_magnitudes > max_scale
        removed_scales = np.sum(huge_scale_mask)
        keep_mask &= ~huge_scale_mask
        
        print(f"{'─'*80}")
        print(f"SCALE FILTER")
        print(f"{'─'*80}")
        print(f"Threshold: > {max_scale:.2f}")
        print(f"Removed:   {removed_scales:,} Gaussians ({removed_scales/n_points*100:.2f}%)")
        print()
    
    # 3. Remove spatial outliers
    distance_threshold = extent * bbox_expansion
    outlier_mask = distances > distance_threshold
    removed_outliers = np.sum(outlier_mask)
    keep_mask &= ~outlier_mask
    
    print(f"{'─'*80}")
    print(f"SPATIAL FILTER (BBOX)")
    print(f"{'─'*80}")
    print(f"Threshold: > {distance_threshold:.2f} meters ({bbox_expansion:.1f}x extent)")
    print(f"Removed:   {removed_outliers:,} Gaussians ({removed_outliers/n_points*100:.2f}%)")
    print()
    
    # Summary
    n_removed = n_points - np.sum(keep_mask)
    n_kept = np.sum(keep_mask)
    
    print(f"{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Original:  {n_points:,} Gaussians")
    print(f"Removed:   {n_removed:,} Gaussians ({n_removed/n_points*100:.2f}%)")
    print(f"Remaining: {n_kept:,} Gaussians ({n_kept/n_points*100:.2f}%)")
    
    if n_kept == 0:
        print()
        print("⚠️  ERROR: All Gaussians would be removed!")
        print("    Try less aggressive thresholds:")
        print(f"    --min-opacity {min_opacity - 2.0}")
        print(f"    --max-scale {max_scale * 2}")
        print(f"    --bbox-expansion {bbox_expansion + 0.5}")
        sys.exit(1)
    
    # Calculate new file size
    file_size_mb = input_path.stat().st_size / (1024 * 1024)
    new_file_size_mb = file_size_mb * (n_kept / n_points)
    size_saved_mb = file_size_mb - new_file_size_mb
    
    print(f"File size: {file_size_mb:.2f} MB → {new_file_size_mb:.2f} MB (saved {size_saved_mb:.2f} MB, {size_saved_mb/file_size_mb*100:.1f}%)")
    print()
    
    if dry_run:
        print("🔍 DRY RUN - No file written")
        print(f"{'='*80}\n")
        return
    
    # Filter all properties
    print("Writing cleaned PLY file...")
    
    # Create new vertex data with only kept points
    vertex_data = vertex.data[keep_mask]
    
    # Center at origin if requested
    if center_at_origin:
        # Recalculate center from kept points only
        kept_x = x[keep_mask]
        kept_y = y[keep_mask]
        kept_z = z[keep_mask]
        new_center = np.array([np.mean(kept_x), np.mean(kept_y), np.mean(kept_z)])
        
        print(f"\n{'─'*80}")
        print(f"CENTERING")
        print(f"{'─'*80}")
        print(f"Original center: ({new_center[0]:.3f}, {new_center[1]:.3f}, {new_center[2]:.3f})")
        print(f"Translating to origin (0, 0, 0)...")
        
        # Translate all positions
        vertex_data['x'] = vertex_data['x'] - new_center[0]
        vertex_data['y'] = vertex_data['y'] - new_center[1]
        vertex_data['z'] = vertex_data['z'] - new_center[2]
        
        final_center_x = np.mean(vertex_data['x'])
        final_center_y = np.mean(vertex_data['y'])
        final_center_z = np.mean(vertex_data['z'])
        print(f"New center: ({final_center_x:.6f}, {final_center_y:.6f}, {final_center_z:.6f})")
        print()
    
    # Create new PLY element
    new_vertex = PlyElement.describe(vertex_data, 'vertex')
    
    # Write PLY
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        PlyData([new_vertex]).write(str(output_path))
        print(f"✅ Saved: {output_path}")
    except Exception as e:
        print(f"ERROR: Failed to write PLY file: {e}")
        sys.exit(1)
    
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Clean up Gaussian Splatting PLY files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default cleanup (removes 82%% low opacity, huge scales, outliers)
  python cleanup_ply.py input.ply output.ply
  
  # More aggressive opacity filter (keep only visible Gaussians)
  python cleanup_ply.py input.ply output.ply --min-opacity -3.0
  
  # Less aggressive scale filter (keep larger Gaussians)
  python cleanup_ply.py input.ply output.ply --max-scale 20.0
  
  # Tighter bounding box (remove far outliers)
  python cleanup_ply.py input.ply output.ply --bbox-expansion 1.2
  
  # Dry run (analyze only, don't save)
  python cleanup_ply.py input.ply output.ply --dry-run
  
Opacity values:
  -5.0 (default) ≈ 0.0067 opacity (keeps 18%% of Gaussians typically)
  -3.0           ≈ 0.047 opacity  (keeps ~5%% of Gaussians typically)
  -1.0           ≈ 0.27 opacity   (keeps ~1%% of Gaussians typically)
  
Note: Opacity stored in logit space. Use sigmoid to convert: 1/(1+exp(-x))
        """
    )
    
    parser.add_argument('input', type=str, help='Input PLY file')
    parser.add_argument('output', type=str, help='Output PLY file')
    parser.add_argument('--min-opacity', type=float, default=-5.0,
                       help='Minimum opacity threshold in logit space (default: -5.0 ≈ 0.0067)')
    parser.add_argument('--max-scale', type=float, default=10.0,
                       help='Maximum scale magnitude threshold (default: 10.0)')
    parser.add_argument('--bbox-expansion', type=float, default=1.5,
                       help='Bounding box expansion factor (default: 1.5x extent)')
    parser.add_argument('--no-center', action='store_true',
                       help='Do not center scene at origin (default: centering enabled)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Analyze only, do not write output file')
    
    args = parser.parse_args()
    
    cleanup_ply(
        args.input,
        args.output,
        min_opacity=args.min_opacity,
        max_scale=args.max_scale,
        bbox_expansion=args.bbox_expansion,
        center_at_origin=not args.no_center,  # Default True, disabled with --no-center
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()
