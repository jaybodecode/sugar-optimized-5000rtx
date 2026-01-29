"""Mesh quality analysis utilities for SUGAR training.

This module provides comprehensive mesh quality metrics including:
- Topology analysis (manifoldness, watertightness, connectivity)
- Triangle quality (area, aspect ratio, degenerates)
- Edge quality (length statistics, uniformity)
- Geometric properties (bounding box, surface area)
- Surface quality (normals, smoothness)

Uses trimesh and open3d libraries for robust mesh analysis.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional, Union
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("Warning: trimesh not available. Install with: pip install trimesh")

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: open3d not available. Install with: pip install open3d")


def compute_mesh_quality_metrics(
    mesh_path: Union[str, Path],
    gaussians_points: Optional[torch.Tensor] = None,
    verbose: bool = True
) -> Dict[str, float]:
    """Compute comprehensive mesh quality metrics.
    
    Args:
        mesh_path: Path to mesh file (.obj, .ply, .stl, etc.)
        gaussians_points: Optional Gaussian point positions for fit analysis
        verbose: Print progress messages
        
    Returns:
        Dictionary of mesh quality metrics
    """
    if not TRIMESH_AVAILABLE:
        return {"error": "trimesh not available"}
    
    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        return {"error": f"Mesh file not found: {mesh_path}"}
    
    if verbose:
        print(f"Analyzing mesh: {mesh_path.name}")
    
    # Load mesh with trimesh
    try:
        mesh = trimesh.load(str(mesh_path), process=False)
    except Exception as e:
        return {"error": f"Failed to load mesh: {e}"}
    
    # Check if it's actually a mesh (not just a point cloud)
    if not hasattr(mesh, 'faces') or len(mesh.faces) == 0:
        return {"error": f"File is not a mesh (no faces): {mesh_path.name}"}
    
    metrics = {}
    
    # ===== Topology Metrics =====
    metrics['n_vertices'] = len(mesh.vertices)
    metrics['n_faces'] = len(mesh.faces)
    metrics['n_edges'] = len(mesh.edges)
    
    # Manifoldness checks
    metrics['is_watertight'] = 1.0 if mesh.is_watertight else 0.0
    metrics['is_winding_consistent'] = 1.0 if mesh.is_winding_consistent else 0.0
    
    # Boundary edges (should be 0 for watertight mesh)
    edge_groups = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
    metrics['n_boundary_edges'] = len(edge_groups)
    
    # Duplicate vertices
    unique_verts = np.unique(mesh.vertices, axis=0)
    metrics['n_duplicate_vertices'] = len(mesh.vertices) - len(unique_verts)
    
    # ===== Triangle Quality Metrics =====
    if hasattr(mesh, 'area_faces') and len(mesh.area_faces) > 0:
        metrics['min_triangle_area'] = float(mesh.area_faces.min())
        metrics['max_triangle_area'] = float(mesh.area_faces.max())
        metrics['avg_triangle_area'] = float(mesh.area_faces.mean())
        metrics['std_triangle_area'] = float(mesh.area_faces.std())
        
        # Count degenerate triangles (very small area)
        degenerate_threshold = 1e-10
        metrics['n_degenerate_triangles'] = int((mesh.area_faces < degenerate_threshold).sum())
    else:
        metrics['min_triangle_area'] = 0.0
        metrics['max_triangle_area'] = 0.0
        metrics['avg_triangle_area'] = 0.0
        metrics['std_triangle_area'] = 0.0
        metrics['n_degenerate_triangles'] = 0
    
    # Triangle aspect ratios (quality measure: 1.0 = equilateral, 0.0 = degenerate)
    try:
        # Compute aspect ratios for all triangles
        v0 = mesh.vertices[mesh.faces[:, 0]]
        v1 = mesh.vertices[mesh.faces[:, 1]]
        v2 = mesh.vertices[mesh.faces[:, 2]]
        
        # Edge lengths
        e0 = np.linalg.norm(v1 - v0, axis=1)
        e1 = np.linalg.norm(v2 - v1, axis=1)
        e2 = np.linalg.norm(v0 - v2, axis=1)
        
        # Semi-perimeter
        s = (e0 + e1 + e2) / 2.0
        
        # Area using Heron's formula
        areas = np.sqrt(np.maximum(s * (s - e0) * (s - e1) * (s - e2), 0))
        
        # Aspect ratio: 4*sqrt(3)*area / (e0^2 + e1^2 + e2^2)
        # This gives 1.0 for equilateral, approaches 0 for degenerate
        perimeter_sq = e0**2 + e1**2 + e2**2
        aspect_ratios = np.where(perimeter_sq > 0, 
                                  4 * np.sqrt(3) * areas / perimeter_sq,
                                  0.0)
        
        metrics['min_aspect_ratio'] = float(aspect_ratios.min())
        metrics['max_aspect_ratio'] = float(aspect_ratios.max())
        metrics['avg_aspect_ratio'] = float(aspect_ratios.mean())
        metrics['std_aspect_ratio'] = float(aspect_ratios.std())
        
        # Count low-quality triangles (aspect ratio < 0.3)
        metrics['n_low_quality_triangles'] = int((aspect_ratios < 0.3).sum())
    except Exception as e:
        if verbose:
            print(f"Warning: Could not compute aspect ratios: {e}")
        metrics['min_aspect_ratio'] = 0.0
        metrics['max_aspect_ratio'] = 0.0
        metrics['avg_aspect_ratio'] = 0.0
        metrics['std_aspect_ratio'] = 0.0
        metrics['n_low_quality_triangles'] = 0
    
    # ===== Edge Quality Metrics =====
    edge_vectors = mesh.vertices[mesh.edges[:, 0]] - mesh.vertices[mesh.edges[:, 1]]
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    
    metrics['min_edge_length'] = float(edge_lengths.min())
    metrics['max_edge_length'] = float(edge_lengths.max())
    metrics['avg_edge_length'] = float(edge_lengths.mean())
    metrics['std_edge_length'] = float(edge_lengths.std())
    
    # Edge length uniformity (coefficient of variation)
    if metrics['avg_edge_length'] > 0:
        metrics['edge_length_uniformity'] = float(
            1.0 - (metrics['std_edge_length'] / metrics['avg_edge_length'])
        )
    else:
        metrics['edge_length_uniformity'] = 0.0
    
    # ===== Geometric Properties =====
    # Bounding box
    metrics['bbox_volume'] = float(mesh.bounding_box.volume)
    metrics['bbox_diagonal'] = float(np.linalg.norm(mesh.bounding_box.extents))
    
    # Store extents separately
    extents = mesh.bounding_box.extents
    metrics['bbox_extent_x'] = float(extents[0])
    metrics['bbox_extent_y'] = float(extents[1])
    metrics['bbox_extent_z'] = float(extents[2])
    
    # Bounding box aspect ratio
    if extents.min() > 0:
        metrics['bbox_aspect_ratio'] = float(extents.max() / extents.min())
    else:
        metrics['bbox_aspect_ratio'] = 1.0
    
    # Surface area
    if hasattr(mesh, 'area'):
        metrics['surface_area'] = float(mesh.area)
    else:
        metrics['surface_area'] = 0.0
    
    # ===== Vertex Connectivity =====
    # Average vertex valence (number of edges per vertex)
    vertex_degrees = np.bincount(mesh.edges.flatten(), minlength=len(mesh.vertices))
    metrics['avg_vertex_valence'] = float(vertex_degrees.mean())
    metrics['max_vertex_valence'] = int(vertex_degrees.max())
    metrics['min_vertex_valence'] = int(vertex_degrees.min())
    
    # Count isolated vertices (valence 0)
    metrics['n_isolated_vertices'] = int((vertex_degrees == 0).sum())
    
    # ===== Surface Normals =====
    if hasattr(mesh, 'face_normals') and len(mesh.face_normals) > 0:
        # Check for degenerate normals (zero length)
        normal_lengths = np.linalg.norm(mesh.face_normals, axis=1)
        metrics['n_degenerate_normals'] = int((normal_lengths < 0.1).sum())
        
        # Normal consistency (neighboring faces should have similar normals)
        try:
            face_adjacency = mesh.face_adjacency
            if len(face_adjacency) > 0:
                # Get normals of adjacent faces
                normals_a = mesh.face_normals[face_adjacency[:, 0]]
                normals_b = mesh.face_normals[face_adjacency[:, 1]]
                
                # Dot product gives cosine of angle between normals
                normal_dots = (normals_a * normals_b).sum(axis=1)
                
                # Average consistency (1.0 = all aligned, -1.0 = all flipped)
                metrics['avg_normal_consistency'] = float(normal_dots.mean())
                
                # Count flipped normals (dot product < 0)
                metrics['n_flipped_normals'] = int((normal_dots < 0).sum())
            else:
                metrics['avg_normal_consistency'] = 1.0
                metrics['n_flipped_normals'] = 0
        except Exception as e:
            if verbose:
                print(f"Warning: Could not compute normal consistency: {e}")
            metrics['avg_normal_consistency'] = 1.0
            metrics['n_flipped_normals'] = 0
    else:
        metrics['n_degenerate_normals'] = 0
        metrics['avg_normal_consistency'] = 1.0
        metrics['n_flipped_normals'] = 0
    
    # ===== Gaussian Fit Analysis =====
    if gaussians_points is not None:
        try:
            # Convert Gaussians to numpy if needed
            if torch.is_tensor(gaussians_points):
                gaussians_np = gaussians_points.detach().cpu().numpy()
            else:
                gaussians_np = np.array(gaussians_points)
            
            # Check which Gaussians are inside the mesh
            if hasattr(mesh, 'contains'):
                inside = mesh.contains(gaussians_np)
                metrics['gaussian_coverage'] = float(inside.mean())
                metrics['n_gaussians_inside'] = int(inside.sum())
                metrics['n_gaussians_outside'] = int((~inside).sum())
            
            # Average distance to mesh surface
            if hasattr(mesh, 'nearest'):
                closest_points, distances, _ = mesh.nearest.on_surface(gaussians_np)
                metrics['avg_dist_to_gaussians'] = float(distances.mean())
                metrics['max_dist_to_gaussians'] = float(distances.max())
                metrics['min_dist_to_gaussians'] = float(distances.min())
        except Exception as e:
            if verbose:
                print(f"Warning: Could not compute Gaussian fit metrics: {e}")
    
    # ===== File Information =====
    metrics['file_size_mb'] = float(mesh_path.stat().st_size / (1024 * 1024))
    
    return metrics


def create_mesh_quality_report(
    metrics: Dict[str, float],
    mesh_path: Optional[Path] = None,
    iteration: Optional[int] = None,
    console: Optional[Console] = None
) -> Panel:
    """Create a Rich console panel with mesh quality summary.
    
    Args:
        metrics: Dictionary of mesh quality metrics
        mesh_path: Optional path to mesh file
        iteration: Optional training iteration number
        console: Optional Rich console for printing
        
    Returns:
        Rich Panel with formatted mesh quality report
    """
    if console is None:
        console = Console()
    
    # Check for error
    if 'error' in metrics:
        return Panel(
            f"[red]Error analyzing mesh:[/red]\n{metrics['error']}",
            title="❌ Mesh Quality Report",
            border_style="red"
        )
    
    # Build report content
    lines = []
    
    if iteration is not None:
        lines.append(f"[bold cyan]Iteration {iteration:,}[/bold cyan]\n")
    
    # Topology section
    lines.append("[bold]📐 Topology[/bold]")
    lines.append(f"  Vertices:            {metrics.get('n_vertices', 0):>10,}")
    lines.append(f"  Faces:               {metrics.get('n_faces', 0):>10,}")
    lines.append(f"  Edges:               {metrics.get('n_edges', 0):>10,}")
    
    is_watertight = metrics.get('is_watertight', 0) > 0.5
    watertight_str = "[green]✓ Yes[/green]" if is_watertight else "[yellow]✗ No[/yellow]"
    lines.append(f"  Watertight:          {watertight_str:>20}")
    
    n_boundary = metrics.get('n_boundary_edges', 0)
    boundary_str = "[green]✓ closed[/green]" if n_boundary == 0 else f"[yellow]{n_boundary} edges[/yellow]"
    lines.append(f"  Boundary Edges:      {boundary_str:>20}")
    
    n_dupes = metrics.get('n_duplicate_vertices', 0)
    dupes_str = "[green]✓ clean[/green]" if n_dupes == 0 else f"[yellow]{n_dupes} vertices[/yellow]"
    lines.append(f"  Duplicate Vertices:  {dupes_str:>20}")
    
    # Triangle quality section
    lines.append("\n[bold]▲ Triangle Quality[/bold]")
    lines.append(f"  Area:       min={metrics.get('min_triangle_area', 0):.5f}  "
                f"avg={metrics.get('avg_triangle_area', 0):.5f}  "
                f"max={metrics.get('max_triangle_area', 0):.3f}")
    
    avg_aspect = metrics.get('avg_aspect_ratio', 0)
    aspect_quality = "[green]✓ good[/green]" if avg_aspect > 0.6 else "[yellow]⚠ poor[/yellow]"
    lines.append(f"  Aspect:     min={metrics.get('min_aspect_ratio', 0):.2f}  "
                f"avg={avg_aspect:.2f}  "
                f"max={metrics.get('max_aspect_ratio', 0):.2f}  {aspect_quality}")
    
    n_degen = metrics.get('n_degenerate_triangles', 0)
    degen_str = "[green]✓ clean[/green]" if n_degen == 0 else f"[red]{n_degen} triangles[/red]"
    lines.append(f"  Degenerate: {degen_str:>20}")
    
    # Edge quality section
    lines.append("\n[bold]─ Edge Quality[/bold]")
    lines.append(f"  Length:     min={metrics.get('min_edge_length', 0):.3f}  "
                f"avg={metrics.get('avg_edge_length', 0):.3f}  "
                f"max={metrics.get('max_edge_length', 0):.3f}")
    
    uniformity = metrics.get('edge_length_uniformity', 0)
    uniformity_pct = uniformity * 100
    uniform_quality = "[green]✓ good[/green]" if uniformity > 0.7 else "[yellow]⚠ varied[/yellow]"
    lines.append(f"  Uniformity: {uniformity_pct:.0f}% {uniform_quality}")
    
    # Bounding box section
    lines.append("\n[bold]📦 Bounding Box[/bold]")
    lines.append(f"  Extents:    {metrics.get('bbox_extent_x', 0):.2f} × "
                f"{metrics.get('bbox_extent_y', 0):.2f} × "
                f"{metrics.get('bbox_extent_z', 0):.2f} m")
    lines.append(f"  Volume:     {metrics.get('bbox_volume', 0):.2f} m³")
    lines.append(f"  Diagonal:   {metrics.get('bbox_diagonal', 0):.2f} m")
    
    # Surface quality section
    lines.append("\n[bold]🎯 Surface Quality[/bold]")
    lines.append(f"  Surface Area:      {metrics.get('surface_area', 0):.2f} m²")
    
    normal_consistency = metrics.get('avg_normal_consistency', 1.0)
    consistency_pct = normal_consistency * 100
    consistency_quality = "[green]✓ smooth[/green]" if normal_consistency > 0.8 else "[yellow]⚠ rough[/yellow]"
    lines.append(f"  Normal Consistency: {consistency_pct:.1f}% {consistency_quality}")
    
    n_flipped = metrics.get('n_flipped_normals', 0)
    flipped_str = "[green]✓ correct[/green]" if n_flipped == 0 else f"[red]{n_flipped} faces[/red]"
    lines.append(f"  Flipped Normals:   {flipped_str:>20}")
    
    avg_valence = metrics.get('avg_vertex_valence', 0)
    valence_quality = "[green]✓ optimal[/green]" if 5.5 <= avg_valence <= 6.5 else "[dim]ok[/dim]"
    lines.append(f"  Avg Vertex Valence: {avg_valence:.1f} {valence_quality}")
    
    # Gaussian fit section (if available)
    if 'gaussian_coverage' in metrics:
        lines.append("\n[bold]✨ Gaussian Fit[/bold]")
        coverage = metrics['gaussian_coverage'] * 100
        coverage_quality = "[green]✓ tight fit[/green]" if coverage > 95 else "[yellow]⚠ loose fit[/yellow]"
        lines.append(f"  Coverage:      {coverage:.1f}% inside mesh {coverage_quality}")
        lines.append(f"  Avg Distance:  {metrics.get('avg_dist_to_gaussians', 0):.4f} m")
        lines.append(f"  Max Distance:  {metrics.get('max_dist_to_gaussians', 0):.4f} m")
    
    # File info section
    if mesh_path is not None:
        lines.append("\n[bold]💾 File Info[/bold]")
        lines.append(f"  Path:   [dim]{mesh_path}[/dim]")
        lines.append(f"  Size:   {metrics.get('file_size_mb', 0):.1f} MB")
    
    # Overall quality assessment
    lines.append("")
    
    # Simple quality scoring
    quality_score = 0
    max_score = 0
    
    # Watertight (20 points)
    if metrics.get('is_watertight', 0) > 0.5:
        quality_score += 20
    max_score += 20
    
    # No boundary edges (10 points)
    if metrics.get('n_boundary_edges', 0) == 0:
        quality_score += 10
    max_score += 10
    
    # Good aspect ratio (20 points)
    if metrics.get('avg_aspect_ratio', 0) > 0.6:
        quality_score += 20
    max_score += 20
    
    # No degenerates (10 points)
    if metrics.get('n_degenerate_triangles', 0) == 0:
        quality_score += 10
    max_score += 10
    
    # Good normal consistency (20 points)
    if metrics.get('avg_normal_consistency', 0) > 0.8:
        quality_score += 20
    max_score += 20
    
    # No flipped normals (10 points)
    if metrics.get('n_flipped_normals', 0) == 0:
        quality_score += 10
    max_score += 10
    
    # Good uniformity (10 points)
    if metrics.get('edge_length_uniformity', 0) > 0.7:
        quality_score += 10
    max_score += 10
    
    quality_pct = (quality_score / max_score) * 100 if max_score > 0 else 0
    
    if quality_pct >= 90:
        quality_label = "[bold green]✅ EXCELLENT[/bold green]"
        quality_desc = "Ready for refinement, Unity import, or direct rendering"
    elif quality_pct >= 75:
        quality_label = "[bold cyan]✓ GOOD[/bold cyan]"
        quality_desc = "Minor issues, should work well in most cases"
    elif quality_pct >= 60:
        quality_label = "[bold yellow]⚠ FAIR[/bold yellow]"
        quality_desc = "Some quality issues, consider cleanup or re-extraction"
    else:
        quality_label = "[bold red]✗ POOR[/bold red]"
        quality_desc = "Significant issues, needs attention"
    
    lines.append(f"{quality_label} [dim]({quality_pct:.0f}/100)[/dim]")
    lines.append(f"[dim]{quality_desc}[/dim]")
    
    content = "\n".join(lines)
    
    return Panel(
        content,
        title="🎨 Mesh Quality Report",
        border_style="cyan",
        expand=False
    )


def log_mesh_metrics_to_tensorboard(
    tb_writer,
    metrics: Dict[str, float],
    iteration: int,
    prefix: str = "Mesh"
):
    """Log mesh quality metrics to TensorBoard with hierarchical naming.
    
    Args:
        tb_writer: TensorBoard SummaryWriter instance
        metrics: Dictionary of mesh quality metrics
        iteration: Training iteration number
        prefix: Prefix for metric names (default: "Mesh")
    """
    if 'error' in metrics:
        return
    
    # Topology metrics
    tb_writer.add_scalar(f'{prefix}/Topology/n_vertices', metrics.get('n_vertices', 0), iteration)
    tb_writer.add_scalar(f'{prefix}/Topology/n_faces', metrics.get('n_faces', 0), iteration)
    tb_writer.add_scalar(f'{prefix}/Topology/n_edges', metrics.get('n_edges', 0), iteration)
    tb_writer.add_scalar(f'{prefix}/Topology/is_watertight', metrics.get('is_watertight', 0), iteration)
    tb_writer.add_scalar(f'{prefix}/Topology/n_boundary_edges', metrics.get('n_boundary_edges', 0), iteration)
    tb_writer.add_scalar(f'{prefix}/Topology/n_duplicate_vertices', metrics.get('n_duplicate_vertices', 0), iteration)
    
    # Triangle quality
    tb_writer.add_scalar(f'{prefix}/Quality/min_triangle_area', metrics.get('min_triangle_area', 0), iteration)
    tb_writer.add_scalar(f'{prefix}/Quality/avg_triangle_area', metrics.get('avg_triangle_area', 0), iteration)
    tb_writer.add_scalar(f'{prefix}/Quality/max_triangle_area', metrics.get('max_triangle_area', 0), iteration)
    tb_writer.add_scalar(f'{prefix}/Quality/min_aspect_ratio', metrics.get('min_aspect_ratio', 0), iteration)
    tb_writer.add_scalar(f'{prefix}/Quality/avg_aspect_ratio', metrics.get('avg_aspect_ratio', 0), iteration)
    tb_writer.add_scalar(f'{prefix}/Quality/max_aspect_ratio', metrics.get('max_aspect_ratio', 0), iteration)
    tb_writer.add_scalar(f'{prefix}/Quality/n_degenerate_triangles', metrics.get('n_degenerate_triangles', 0), iteration)
    tb_writer.add_scalar(f'{prefix}/Quality/n_low_quality_triangles', metrics.get('n_low_quality_triangles', 0), iteration)
    
    # Edge quality
    tb_writer.add_scalar(f'{prefix}/Quality/min_edge_length', metrics.get('min_edge_length', 0), iteration)
    tb_writer.add_scalar(f'{prefix}/Quality/avg_edge_length', metrics.get('avg_edge_length', 0), iteration)
    tb_writer.add_scalar(f'{prefix}/Quality/max_edge_length', metrics.get('max_edge_length', 0), iteration)
    tb_writer.add_scalar(f'{prefix}/Quality/edge_length_uniformity', metrics.get('edge_length_uniformity', 0), iteration)
    
    # Geometry
    tb_writer.add_scalar(f'{prefix}/Geometry/bbox_volume', metrics.get('bbox_volume', 0), iteration)
    tb_writer.add_scalar(f'{prefix}/Geometry/bbox_diagonal', metrics.get('bbox_diagonal', 0), iteration)
    tb_writer.add_scalar(f'{prefix}/Geometry/surface_area', metrics.get('surface_area', 0), iteration)
    tb_writer.add_scalar(f'{prefix}/Geometry/bbox_aspect_ratio', metrics.get('bbox_aspect_ratio', 1), iteration)
    
    # Surface quality
    tb_writer.add_scalar(f'{prefix}/Surface/avg_normal_consistency', metrics.get('avg_normal_consistency', 1), iteration)
    tb_writer.add_scalar(f'{prefix}/Surface/n_flipped_normals', metrics.get('n_flipped_normals', 0), iteration)
    tb_writer.add_scalar(f'{prefix}/Surface/n_degenerate_normals', metrics.get('n_degenerate_normals', 0), iteration)
    tb_writer.add_scalar(f'{prefix}/Surface/avg_vertex_valence', metrics.get('avg_vertex_valence', 0), iteration)
    
    # Gaussian fit (if available)
    if 'gaussian_coverage' in metrics:
        tb_writer.add_scalar(f'{prefix}/GaussianFit/coverage', metrics.get('gaussian_coverage', 0), iteration)
        tb_writer.add_scalar(f'{prefix}/GaussianFit/avg_distance', metrics.get('avg_dist_to_gaussians', 0), iteration)
        tb_writer.add_scalar(f'{prefix}/GaussianFit/max_distance', metrics.get('max_dist_to_gaussians', 0), iteration)
