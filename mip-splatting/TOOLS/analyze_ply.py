#!/usr/bin/env python3
"""
Standalone PLY Analysis Tool for Gaussian Splatting Models

Analyzes any .ply file and generates a comprehensive report including:
- Gaussian statistics (count, bounds, distributions)
- Custom attributes (3D filter, scales, opacity)
- Quality metrics (if provided: PSNR, SSIM, LPIPS)
- Potential defects detection

Usage:
    python analyze_ply.py point_cloud.ply
    python analyze_ply.py point_cloud.ply --psnr 28.5 --ssim 0.92 --lpips 0.043
    python analyze_ply.py point_cloud.ply --iteration 7000 --loss 0.0247
"""

import argparse
import sys
from pathlib import Path
import numpy as np

try:
    from plyfile import PlyData
except ImportError:
    print("ERROR: plyfile not installed. Run: pip install plyfile")
    sys.exit(1)


def analyze_ply(ply_path, metadata=None):
    """Analyze a PLY file and return comprehensive statistics"""
    
    ply_path = Path(ply_path)
    if not ply_path.exists():
        print(f"ERROR: File not found: {ply_path}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"PLY ANALYSIS REPORT")
    print(f"{'='*80}\n")
    
    # Load PLY
    print(f"📁 File: {ply_path.name}")
    print(f"📂 Path: {ply_path.parent}")
    file_size_mb = ply_path.stat().st_size / (1024 * 1024)
    print(f"💾 Size: {file_size_mb:.2f} MB")
    print()
    
    try:
        plydata = PlyData.read(str(ply_path))
    except Exception as e:
        print(f"ERROR: Failed to read PLY file: {e}")
        sys.exit(1)
    
    vertex = plydata['vertex']
    n_points = len(vertex)
    
    print(f"{'─'*80}")
    print(f"MODEL STATISTICS")
    print(f"{'─'*80}\n")
    
    # Basic counts
    print(f"🎯 Gaussian Count: {n_points:,} points")
    print(f"📊 Bytes per Point: {file_size_mb * 1024 * 1024 / n_points:.1f} bytes")
    print()
    
    # Extract positions
    x = vertex['x']
    y = vertex['y']
    z = vertex['z']
    
    # Bounding box
    print(f"{'─'*80}")
    print(f"SPATIAL BOUNDS")
    print(f"{'─'*80}\n")
    
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    min_z, max_z = np.min(z), np.max(z)
    
    print(f"X: [{min_x:+.3f}, {max_x:+.3f}]  (range: {max_x - min_x:.3f})")
    print(f"Y: [{min_y:+.3f}, {max_y:+.3f}]  (range: {max_y - min_y:.3f})")
    print(f"Z: [{min_z:+.3f}, {max_z:+.3f}]  (range: {max_z - min_z:.3f})")
    print()
    
    # Center and extent
    center = np.array([np.mean(x), np.mean(y), np.mean(z)])
    print(f"📍 Center: ({center[0]:+.3f}, {center[1]:+.3f}, {center[2]:+.3f})")
    
    distances = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
    extent = np.max(distances)
    print(f"📏 Extent (radius): {extent:.3f} meters")
    print()
    
    # Furthest points
    furthest_idx = np.argmax(distances)
    print(f"🔭 Furthest Point: ({x[furthest_idx]:+.3f}, {y[furthest_idx]:+.3f}, {z[furthest_idx]:+.3f})")
    print(f"   Distance from center: {distances[furthest_idx]:.3f} meters")
    print()
    
    # Available properties
    print(f"{'─'*80}")
    print(f"PLY PROPERTIES")
    print(f"{'─'*80}\n")
    
    properties = vertex.data.dtype.names
    print(f"Available attributes ({len(properties)}):")
    for prop in properties:
        print(f"  • {prop}")
    print()
    
    # Analyze custom attributes
    print(f"{'─'*80}")
    print(f"GAUSSIAN ATTRIBUTES")
    print(f"{'─'*80}\n")
    
    # Opacity
    if 'opacity' in properties:
        opacity = vertex['opacity']
        print(f"Opacity:")
        print(f"  Range: [{np.min(opacity):.6f}, {np.max(opacity):.6f}]")
        print(f"  Mean:  {np.mean(opacity):.6f}")
        print(f"  Std:   {np.std(opacity):.6f}")
        
        # Detect very low opacity (potential culling candidates)
        low_opacity = np.sum(opacity < 0.01)
        if low_opacity > 0:
            print(f"  ⚠️  Low opacity (<0.01): {low_opacity:,} points ({low_opacity/n_points*100:.2f}%)")
        print()
    
    # Scales
    scale_props = [p for p in properties if p.startswith('scale_')]
    if scale_props:
        scales = np.array([vertex[p] for p in scale_props]).T  # Shape: (n_points, 3)
        scale_magnitudes = np.linalg.norm(scales, axis=1)
        
        print(f"Scales (3D):")
        print(f"  Magnitude range: [{np.min(scale_magnitudes):.6f}, {np.max(scale_magnitudes):.6f}]")
        print(f"  Mean magnitude:  {np.mean(scale_magnitudes):.6f}")
        
        # Detect huge scales (potential artifacts)
        huge_scales = np.sum(scale_magnitudes > 10.0)
        if huge_scales > 0:
            print(f"  ⚠️  Huge scales (>10.0): {huge_scales:,} points ({huge_scales/n_points*100:.2f}%)")
        
        # Detect tiny scales (potential noise)
        tiny_scales = np.sum(scale_magnitudes < 0.0001)
        if tiny_scales > 0:
            print(f"  ⚠️  Tiny scales (<0.0001): {tiny_scales:,} points ({tiny_scales/n_points*100:.2f}%)")
        print()
    
    # 3D filter (Mip-Splatting custom attribute)
    if 'filter_3D' in properties:
        filter_3d = vertex['filter_3D']
        print(f"3D Filter (Mip-Splatting):")
        print(f"  Range: [{np.min(filter_3d):.6f}, {np.max(filter_3d):.6f}]")
        print(f"  Mean:  {np.mean(filter_3d):.6f}")
        print(f"  Std:   {np.std(filter_3d):.6f}")
        print()
    
    # Colors (SH coefficients)
    sh_props = [p for p in properties if p.startswith('f_dc_') or p.startswith('f_rest_')]
    if sh_props:
        print(f"Spherical Harmonics: {len(sh_props)} coefficients")
        
        # DC component (base color)
        dc_props = [p for p in properties if p.startswith('f_dc_')]
        if dc_props:
            dc_values = np.array([vertex[p] for p in dc_props]).T
            print(f"  DC (base RGB): {len(dc_props)} channels")
            print(f"    Range: [{np.min(dc_values):.6f}, {np.max(dc_values):.6f}]")
        print()
    
    # Quality Metrics (if provided)
    if metadata:
        print(f"{'─'*80}")
        print(f"QUALITY METRICS")
        print(f"{'─'*80}\n")
        
        if 'iteration' in metadata:
            print(f"Iteration: {metadata['iteration']}")
        
        if 'loss' in metadata:
            print(f"Training Loss: {metadata['loss']:.6f}")
        
        if 'psnr' in metadata:
            print(f"PSNR:  {metadata['psnr']:.2f} dB")
        
        if 'ssim' in metadata:
            print(f"SSIM:  {metadata['ssim']:.4f}")
        
        if 'lpips' in metadata:
            print(f"LPIPS: {metadata['lpips']:.4f}")
        
        print()
    
    # Defect Detection
    print(f"{'─'*80}")
    print(f"DEFECT ANALYSIS")
    print(f"{'─'*80}\n")
    
    issues = []
    
    # Check for NaN/Inf values
    if np.any(np.isnan(x)) or np.any(np.isnan(y)) or np.any(np.isnan(z)):
        nan_count = np.sum(np.isnan(x) | np.isnan(y) | np.isnan(z))
        issues.append(f"NaN positions: {nan_count:,} points")
    
    if np.any(np.isinf(x)) or np.any(np.isinf(y)) or np.any(np.isinf(z)):
        inf_count = np.sum(np.isinf(x) | np.isinf(y) | np.isinf(z))
        issues.append(f"Infinite positions: {inf_count:,} points")
    
    # Check for outliers (>3 sigma from center)
    distance_threshold = extent * 1.5
    outliers = np.sum(distances > distance_threshold)
    if outliers > n_points * 0.01:  # More than 1% outliers
        issues.append(f"Spatial outliers: {outliers:,} points ({outliers/n_points*100:.2f}%) beyond 1.5x extent")
    
    # Check opacity distribution
    if 'opacity' in properties:
        very_low_opacity = np.sum(opacity < 0.001)
        if very_low_opacity > n_points * 0.1:  # More than 10%
            issues.append(f"Many invisible Gaussians: {very_low_opacity:,} points ({very_low_opacity/n_points*100:.2f}%) with opacity < 0.001")
    
    if issues:
        print("⚠️  Potential Issues Detected:")
        for issue in issues:
            print(f"  • {issue}")
    else:
        print("✅ No obvious defects detected")
    
    print()
    print(f"{'='*80}\n")
    
    return {
        'n_points': n_points,
        'file_size_mb': file_size_mb,
        'bounds': {
            'x': (min_x, max_x),
            'y': (min_y, max_y),
            'z': (min_z, max_z),
        },
        'center': center.tolist(),
        'extent': extent,
        'properties': list(properties),
        'issues': issues,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Gaussian Splatting PLY files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_ply.py point_cloud.ply
  python analyze_ply.py point_cloud.ply --psnr 28.5 --ssim 0.92 --lpips 0.043
  python analyze_ply.py point_cloud.ply --iteration 7000 --loss 0.0247
        """
    )
    
    parser.add_argument('ply_file', type=str, help='Path to PLY file')
    parser.add_argument('--iteration', type=int, help='Training iteration number')
    parser.add_argument('--loss', type=float, help='Training loss value')
    parser.add_argument('--psnr', type=float, help='PSNR value (dB)')
    parser.add_argument('--ssim', type=float, help='SSIM value')
    parser.add_argument('--lpips', type=float, help='LPIPS value')
    
    args = parser.parse_args()
    
    # Build metadata dict
    metadata = {}
    if args.iteration is not None:
        metadata['iteration'] = args.iteration
    if args.loss is not None:
        metadata['loss'] = args.loss
    if args.psnr is not None:
        metadata['psnr'] = args.psnr
    if args.ssim is not None:
        metadata['ssim'] = args.ssim
    if args.lpips is not None:
        metadata['lpips'] = args.lpips
    
    analyze_ply(args.ply_file, metadata if metadata else None)


if __name__ == '__main__':
    main()
