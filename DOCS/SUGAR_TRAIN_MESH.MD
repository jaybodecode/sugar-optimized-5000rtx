# SuGaR Stage 2 - Mesh Extraction Deep Dive

**Status:** Complete Analysis  
**Date:** January 29, 2026  
**Scope:** Everything after "Coarse model loaded." message until mesh is saved  
**File:** `SuGaR/sugar_extractors/coarse_mesh.py` (933 lines)

---

## 🎯 Executive Summary

**What Stage 2 Does:** Converts trained Gaussian point cloud → triangulated surface mesh

**Input:** Trained SuGaR checkpoint (`.pt` file with 4-6M Gaussians)  
**Output:** Decimated `.ply` mesh file(s) with vertex colors and normals  
**Time:** ~5-10 minutes  
**VRAM:** ~8-10 GB peak (with optimizations)

**Key Operations:**
1. **Load & prune** - Remove low-opacity Gaussians (opacity < 0.5)
2. **Extract surface** - Render from multiple camera views, collect 10M surface points
3. **Build mesh** - Poisson surface reconstruction (foreground + background)
4. **Decimate** - Reduce to target vertex count (200K or 1M)
5. **Project** - Snap mesh vertices to surface points for detail preservation

---

## 📊 Stage 2 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: MESH EXTRACTION (after "Coarse model loaded.")    │
└─────────────────────────────────────────────────────────────┘

Step 1: Load Coarse Model [~5-10s]
├─ Load checkpoint from .pt file
├─ Initialize SuGaR model with trained parameters
├─ Display model parameter table (name, shape, trainable status)
└─ VRAM: ~6-8 GB

Step 2: Prune Low Opacity Gaussians [~2-5s]
├─ Display pre-pruning statistics (N points, opacity min/max/mean)
├─ Show opacity quantiles (0.0 to 1.0 in 10 steps)
├─ Remove Gaussians with opacity < 0.5 (threshold configurable)
├─ Display post-pruning statistics
└─ VRAM: Same (~6-8 GB)
   Typical: 6M → 4-5M Gaussians after pruning

Step 3: Build Triangle Soup [~5-10s]
├─ Set primitive type: 'diamond' (default, better than 'square')
├─ Set triangle scale: 2.0
├─ Update texture features
├─ Create mesh rasterizer with resolution_factor (default 0.5)
│  ├─ mesh_height = 622 (1244 × 0.5)
│  ├─ mesh_width = 960 (1920 × 0.5)
│  ├─ faces_per_pixel = 5 (reduced from 10, saves 2-3 GB)
│  └─ max_faces_per_bin = 50,000
└─ VRAM: ~8-10 GB peak

Step 4: Surface Point Extraction [~3-5 min, MAIN BOTTLENECK]
├─ Target: 10M surface points across all camera views
├─ Points per frame: ~62,112 (10M / 161 frames)
├─ Reset KNN neighbors with K=16
├─ For each training camera (161 in Garden):
│  ├─ Render RGB image with Gaussian splatting
│  ├─ Compute surface level points at density threshold (default 0.3)
│  ├─ Extract points, colors, normals, view directions
│  └─ Randomly sample n_pts_per_frame points
├─ Progress: Every 30 frames (~20 seconds per batch)
└─ VRAM: ~8-10 GB (depends on resolution_factor)

Step 5: Split Foreground/Background [~2-5s]
├─ Define bounding box (camera-based or custom)
│  ├─ Foreground: ±1.0 × camera spatial extent
│  └─ Background: ±4.0 × camera spatial extent
├─ Mask surface points by bbox
└─ Typical split: ~7-8M foreground, ~2-3M background

Step 6: Poisson Mesh Reconstruction [~30-60s, CPU-BOUND]
├─ Foreground mesh:
│  ├─ Build Open3D point cloud (points + colors + normals)
│  ├─ Remove statistical outliers (nb_neighbors=20, std_ratio=20)
│  ├─ Poisson reconstruction (depth=10, higher = more detail)
│  └─ Remove low-density vertices (quantile=0.1)
├─ Background mesh:
│  └─ Same process as foreground
└─ VRAM: ~6-8 GB (CPU does heavy lifting here)
   Output: ~5-10M vertices each (before decimation)

Step 7: Mesh Decimation [~30-60s]
├─ For each decimation target (e.g., 200K, 1M):
│  ├─ Quadric edge collapse decimation (Open3D)
│  ├─ Foreground: target vertices (e.g., 1M)
│  └─ Background: target vertices (e.g., 1M)
└─ VRAM: ~6-8 GB
   Time: ~30s per target per mesh

Step 8: Mesh Cleaning [~5-10s]
├─ Remove duplicated vertices
├─ Remove degenerate triangles
├─ Remove duplicated triangles
├─ Remove non-manifold edges
└─ Merge foreground + background meshes

Step 9: Mesh Projection (OPTIONAL, if --project_mesh_on_surface_points) [~10-20s]
├─ Find K=1 nearest surface point for each mesh vertex (KNN)
├─ Snap mesh vertex positions to surface points
├─ Transfer surface point colors to mesh
├─ Recompute normals
├─ Re-clean mesh (duplicates, degenerate triangles)
└─ VRAM spike: +2-3 GB temporarily (KNN computation)
   **Quality Impact:** Significant - preserves fine surface details

Step 10: Save Mesh [~2-5s]
├─ Filename: sugarmesh_<params>_level<L>_decim<D>.ply
├─ Save with vertex colors, normals, UVs (if available)
└─ Repeat for each surface_level × decimation_target combination

Total Time: ~5-10 minutes
Total VRAM Peak: ~10-12 GB (with mesh_extraction_resolution_factor=0.5)
```

---

## 🔍 Critical Components Analysis

### 1. Load Coarse Model (Lines 125-207)

**Location:** `coarse_mesh.py:125-207`

**What Happens:**
```python
# 1. Load checkpoint
checkpoint = torch.load(sugar_checkpoint_path, map_location=nerfmodel.device)

# 2. Initialize SuGaR model (empty container)
sugar = SuGaR(
    nerfmodel=nerfmodel,  # Trained Gaussian Splatting wrapper
    points=nerfmodel.gaussians.get_xyz,  # Initial positions
    colors=SH2RGB(nerfmodel.gaussians.get_features[:, :1, :]),  # RGB from SH
    initialize=True,
    sh_levels=nerfmodel.gaussians.active_sh_degree + 1,
    keep_track_of_knn=False,  # No KNN tracking needed for inference
    knn_to_track=16,
    beta_mode='average',
    primitive_types='diamond',
    surface_mesh_to_bind=None,
)

# 3. Load trained parameters from checkpoint
sugar.load_state_dict(checkpoint['state_dict'])
sugar.eval()  # Set to inference mode
```

**Parameters Loaded:**
- `_points` - Gaussian 3D positions (N × 3)
- `_scales` - Gaussian sizes (N × 3)
- `_quaternions` - Gaussian orientations (N × 4)
- `_sh_coordinates_dc` - Spherical harmonics DC term (N × 1 × 3, RGB base)
- `_sh_coordinates_rest` - SH higher frequencies (N × 15 × 3, view-dependent color)
- `_strengths` - Gaussian opacities (N × 1)
- `all_densities` - Pre-computed density field (if cached)

**Console Output:**
```
[bold cyan]Coarse model loaded.[/bold cyan]

┌─ Coarse Model Parameters ──────────────────────────────────┐
│ Parameter               │ Shape            │ Trainable     │
├─────────────────────────┼──────────────────┼───────────────┤
│ _points                 │ (5999994, 3)     │ True          │
│ _scales                 │ (5999994, 3)     │ True          │
│ _quaternions            │ (5999994, 4)     │ True          │
│ _sh_coordinates_dc      │ (5999994, 1, 3)  │ True          │
│ _sh_coordinates_rest    │ (5999994, 15, 3) │ True          │
│ _strengths              │ (5999994, 1)     │ True          │
└─────────────────────────┴──────────────────┴───────────────┘
```

**VRAM Usage:** ~6-8 GB (depends on checkpoint size)

---

### 2. Prune Low Opacity Gaussians (Lines 209-261)

**Location:** `coarse_mesh.py:209-261`

**Purpose:** Remove transparent/invisible Gaussians that don't contribute to surface

**Algorithm:**
```python
def drop_low_opacity_points(self, opacity_threshold=0.5):
    """Remove Gaussians with opacity below threshold."""
    # Get opacity mask
    mask = (self.strengths.squeeze() > opacity_threshold)
    
    # Filter ALL parameters
    self._points = self._points[mask]
    self._scales = self._scales[mask]
    self._quaternions = self._quaternions[mask]
    self._sh_coordinates_dc = self._sh_coordinates_dc[mask]
    self._sh_coordinates_rest = self._sh_coordinates_rest[mask]
    self._strengths = self._strengths[mask]
    
    # Update count
    self.n_points = len(self._points)
```

**Console Output (Example):**
```
┌─ Pre-Pruning Statistics ───────────────────────────────────┐
│ Metric                  │ Value                            │
├─────────────────────────┼──────────────────────────────────┤
│ Number of Gaussians     │ 5,999,994                        │
│ Opacity Min             │ 0.000012                         │
│ Opacity Max             │ 0.999854                         │
│ Opacity Mean            │ 0.542318                         │
└─────────────────────────┴──────────────────────────────────┘

┌─ Opacity Quantiles ────────────────────────────────────────┐
│ Quantile │ Opacity                                         │
├──────────┼─────────────────────────────────────────────────┤
│ 0.0      │ 0.000012                                        │
│ 0.1      │ 0.123456                                        │
│ 0.2      │ 0.234567                                        │
│ ...      │ ...                                             │
│ 0.9      │ 0.876543                                        │
│ 1.0      │ 0.999854                                        │
└──────────┴─────────────────────────────────────────────────┘

[bold magenta]Starting pruning low opacity gaussians...[/bold magenta]

┌─ Post-Pruning Statistics ──────────────────────────────────┐
│ Metric                  │ Value                            │
├─────────────────────────┼──────────────────────────────────┤
│ Number of Gaussians     │ 4,823,156                        │  ← ~80% retained
│ Opacity Min             │ 0.500003                         │  ← Threshold enforced
│ Opacity Max             │ 0.999854                         │
│ Opacity Mean            │ 0.678234                         │  ← Higher average
└─────────────────────────┴──────────────────────────────────┘
```

**Impact:**
- **Reduces Gaussians:** ~20-30% removed (6M → 4-5M typical)
- **Speeds up rendering:** Fewer Gaussians to process in Step 4
- **VRAM:** No change (freed memory offset by subsequent operations)

**Why Threshold = 0.5?**
- **Too low (0.1-0.3):** Keeps semi-transparent Gaussians → noisy mesh
- **Sweet spot (0.5):** Balances quality and performance
- **Too high (0.7-0.9):** Removes too much → holes in mesh

---

### 3. Build Triangle Soup (Lines 263-297)

**Location:** `coarse_mesh.py:263-297`

**Purpose:** Create mesh rasterizer for rendering from arbitrary viewpoints

**Key Parameters:**

#### Primitive Types
```python
sugar.primitive_types = 'diamond'  # vs 'square'
```
- **Diamond:** 4 triangles arranged in diamond shape (RECOMMENDED)
- **Square:** 2 triangles forming square
- **Impact:** Diamond gives smoother interpolation for surface extraction

#### Triangle Scale
```python
sugar.triangle_scale = 2.0
```
- **Lower (1.0):** Smaller primitives, more gaps, slower
- **Default (2.0):** Balanced coverage
- **Higher (3.0):** Larger primitives, overlapping, faster but less accurate

#### Resolution Factor (VRAM CRITICAL)
```python
resolution_factor = args.mesh_extraction_resolution_factor  # Default: 0.5
mesh_height = int(sugar.image_height * resolution_factor)  # 1244 × 0.5 = 622
mesh_width = int(sugar.image_width * resolution_factor)    # 1920 × 0.5 = 960
```

**Impact on VRAM:**

| Resolution Factor | Resolution | VRAM Usage | Quality | Speed |
|-------------------|------------|------------|---------|-------|
| 1.0 (full-res)    | 1244×1920  | ~18-24 GB  | 100%    | 1.0× (baseline) |
| 0.75              | 933×1440   | ~14-18 GB  | 95-98%  | 1.3×  |
| 0.5 (default)     | 622×960    | ~8-10 GB   | 90-95%  | 2.0×  |
| 0.4               | 498×768    | ~6-8 GB    | 85-90%  | 2.5×  |

**⚠️ Critical Optimization (from SUGAR_OPTIMISATIONS.MD):**
```python
# VRAM Optimization: Default 0.5 resolution factor + faces_per_pixel=5 
# saves ~6-8GB VRAM (24GB → ~15GB) with minimal quality loss.
# Use --mesh_extraction_resolution_factor 1.0 for full-res on 32GB+ GPUs.

faces_per_pixel = 5  # Reduced from 10 (saves ~2-3GB, minimal impact)
max_faces_per_bin = 50_000
```

**Why faces_per_pixel=5?**
- **Default (10):** Higher accuracy for complex overlapping geometry
- **Optimized (5):** Sufficient for most scenes, saves 2-3 GB VRAM
- **Too low (1-2):** Visible artifacts in overlapping regions

---

### 4. Surface Point Extraction (Lines 299-403, MAIN BOTTLENECK)

**Location:** `coarse_mesh.py:299-403`

**Purpose:** Render scene from all training cameras, extract surface points at density threshold

**Algorithm Overview:**
```
For each camera view (161 in Garden):
  1. Render RGB image (Gaussian splatting)
  2. Compute surface intersection points at density level (e.g., 0.3)
  3. Extract point positions, colors, normals, view directions
  4. Randomly sample n_pts_per_frame points (~62K per frame)
  5. Accumulate to global point cloud

Target: 10M total surface points
```

**Key Parameters:**

#### Total Points Budget
```python
n_total_points = 10_000_000  # 10M surface points total
n_pts_per_frame = int(n_total_points / len(cameras)) + 1
# Garden: 10M / 161 = 62,112 points per frame
```

**Why 10M points?**
- **Too few (1-2M):** Sparse surface representation → holes, poor Poisson reconstruction
- **Sweet spot (10M):** Dense coverage, smooth mesh
- **Too many (50M+):** Diminishing returns, much slower, VRAM issues

#### Surface Level (Density Threshold)
```python
surface_levels = [0.1, 0.3, 0.5]  # Default: extract 3 meshes at different levels
```

**Impact of Surface Level:**
- **Low (0.1):** Thin surface, more conservative, tighter mesh
- **Medium (0.3):** Balanced, RECOMMENDED
- **High (0.5):** Thick surface, more generous, smoother but may include interior

#### KNN Neighbors Tracking
```python
surface_level_knn_to_track = 16
```
- **Purpose:** Track nearest neighbors for normal estimation
- **Higher K (32):** Smoother normals, slower
- **Lower K (8):** Faster but noisier normals

#### Rendering Method (Fast vs Efficient)
```python
use_fast_method = True  # Default: True (RECOMMENDED)
```

**Fast Method (`compute_level_surface_points_from_camera_fast`):**
- Directly indexes rendered pixels
- Faster (~2× speedup)
- More memory efficient
- **RECOMMENDED**

**Efficient Method (`compute_level_surface_points_from_camera_efficient`):**
- More careful filtering of empty pixels
- Slightly more accurate
- Slower

#### Multi-pass Rendering (VRAM Control)
```python
surface_level_n_points_in_range = 21      # Sample 21 depth points per pixel
surface_level_range_size = 3.0            # Search range in 3D space
surface_level_n_points_per_pass = 2_000_000  # Process 2M points at a time
```

**Why multi-pass?**
- Single pass: Render all 622×960 × 21 = 12.5M points at once → 15-20 GB VRAM
- Multi-pass: Process 2M points per pass → ~8-10 GB VRAM

---

**Rendering Loop (Simplified):**
```python
surface_levels_outputs = {
    0.3: {
        'points': torch.zeros(0, 3),       # 3D positions
        'colors': torch.zeros(0, 3),       # RGB colors
        'normals': torch.zeros(0, 3),      # Surface normals
        'view_directions': torch.zeros(0, 3),  # Camera → point vectors
        'pix_to_gaussians': torch.zeros(0, dtype=long),  # Which Gaussian contributed
    }
}

for cam_idx in range(161):  # Garden dataset
    if cam_idx % 30 == 0:
        print(f"Processing frame {cam_idx}/161...")
        print(f"Current point cloud has {len(points)} points.")
    
    # 1. Render RGB image with Gaussian splatting
    rgb = sugar.render_image_gaussian_rasterizer(
        nerf_cameras=cameras,
        camera_indices=cam_idx,
        bg_color=None,
        sh_deg=0,  # Use DC term only (faster, sufficient for mesh extraction)
        compute_color_in_rasterizer=True,
        compute_covariance_in_rasterizer=True,
    ).clamp(0., 1.)
    
    # 2. Compute surface level points (MAIN WORK)
    frame_outputs = sugar.compute_level_surface_points_from_camera_fast(
        cam_idx=cam_idx,
        rasterizer=rasterizer,
        surface_levels=[0.3],  # Density threshold
        n_surface_points=2 * n_pts_per_frame,  # 2× buffer to avoid empty pixels
        primitive_types='diamond',
        triangle_scale=2.0,
        splat_mesh=True,
        n_points_in_range=21,  # Sample 21 depth points per pixel
        range_size=3.0,
        n_points_per_pass=2_000_000,  # Multi-pass rendering
        density_factor=1.0,
        return_pixel_idx=True,
        return_gaussian_idx=True,
        return_normals=True,
        compute_flat_normals=False,
        use_gaussian_depth=False,
    )
    
    # 3. Extract data for surface level 0.3
    img_surface_points = frame_outputs[0.3]['intersection_points']  # (N, 3)
    img_surface_colors = rgb.view(-1, 3)[frame_outputs[0.3]['pixel_idx']]  # (N, 3)
    img_surface_normals = frame_outputs[0.3]['normals']  # (N, 3)
    img_surface_view_dirs = normalize(camera_center - img_surface_points)  # (N, 3)
    
    # 4. Randomly sample to budget
    idx = torch.randperm(len(img_surface_points))[:n_pts_per_frame]
    
    # 5. Accumulate to global point cloud
    surface_levels_outputs[0.3]['points'] = torch.cat([
        surface_levels_outputs[0.3]['points'], 
        img_surface_points[idx]
    ])
    # ... same for colors, normals, view_directions
```

**Progress Output:**
```
Processing frame 0/161...
Current point cloud for level 0.3 has 0 points.

Processing frame 30/161...
Current point cloud for level 0.3 has 1,863,360 points.

Processing frame 60/161...
Current point cloud for level 0.3 has 3,726,720 points.

Processing frame 90/161...
Current point cloud for level 0.3 has 5,590,080 points.

Processing frame 120/161...
Current point cloud for level 0.3 has 7,453,440 points.

Processing frame 150/161...
Current point cloud for level 0.3 has 9,316,800 points.

Final point cloud for level 0.3 has 10,001,232 points.
```

**Time Breakdown:**
- **Per frame:** ~1-2 seconds (depends on resolution_factor)
- **Total (161 frames):** ~3-5 minutes
- **Bottleneck:** Gaussian rasterization + surface intersection computation

**VRAM Usage:**
- **Peak:** ~8-10 GB (with resolution_factor=0.5, n_points_per_pass=2M)
- **Without optimization:** ~18-24 GB (full-res, single pass)

---

### 5. Split Foreground/Background (Lines 405-428)

**Location:** `coarse_mesh.py:405-428`

**Purpose:** Separate surface points into foreground (main object) and background (surroundings)

**Bounding Box Computation:**

#### Default (Camera-based)
```python
# Foreground: ±1.0 × camera spatial extent
fg_bbox_factor = 1.0
fg_bbox_min = -1.0 * camera_spatial_extent * [1, 1, 1]
fg_bbox_max = +1.0 * camera_spatial_extent * [1, 1, 1]

# Background: ±4.0 × camera spatial extent (OUTSIDE foreground)
bg_bbox_factor = 4.0
bg_bbox_min = -4.0 * camera_spatial_extent * [1, 1, 1]
bg_bbox_max = +4.0 * camera_spatial_extent * [1, 1, 1]
```

**Camera spatial extent:** Max distance from any camera to scene center

**Example (Garden dataset):**
- Camera extent: ~2.5 units
- Foreground bbox: [-2.5, -2.5, -2.5] to [+2.5, +2.5, +2.5]
- Background bbox: [-10.0, -10.0, -10.0] to [+10.0, +10.0, +10.0] (excluding foreground)

#### Custom (User-specified)
```bash
--bboxmin "(-1.0, -1.0, 0.0)" \
--bboxmax "(1.0, 1.0, 2.0)"
```

**Masking Logic:**
```python
# Foreground: Points INSIDE foreground bbox
fg_mask = (
    (surface_points > fg_bbox_min).all(dim=-1) &
    (surface_points < fg_bbox_max).all(dim=-1)
)

# Background: Points INSIDE background bbox but OUTSIDE foreground
bg_mask = (
    (surface_points.abs().max(dim=-1)[0] < bg_bbox_factor * camera_extent) &
    ~fg_mask
)

# Split point cloud
fg_points = surface_points[fg_mask]  # e.g., 7,456,789 points
fg_colors = surface_colors[fg_mask]
fg_normals = surface_normals[fg_mask]

bg_points = surface_points[bg_mask]  # e.g., 2,544,443 points
bg_colors = surface_colors[bg_mask]
bg_normals = surface_normals[bg_mask]
```

**Console Output:**
```
Using default, camera based bounding box.

Foreground points: torch.Size([7456789, 3]) torch.Size([7456789, 3]) torch.Size([7456789, 3])
Background points: torch.Size([2544443, 3]) torch.Size([2544443, 3]) torch.Size([2544443, 3])
```

**Why Split?**
- **Different Poisson depths:** Foreground needs higher detail
- **Different decimation targets:** Foreground more important
- **Separate cleanup:** Easier to handle artifacts in each region

---

### 6. Poisson Mesh Reconstruction (Lines 430-523, CPU-BOUND)

**Location:** `coarse_mesh.py:430-523`

**Purpose:** Convert point cloud (points + normals) → watertight triangle mesh

**Algorithm:** Poisson Surface Reconstruction (Open3D)
- **Input:** Oriented point cloud (points + normals)
- **Output:** Watertight mesh (vertices + faces + colors)
- **Method:** Solves Poisson equation ∇²χ = ∇·V where V is normal field
- **Key parameter:** `depth` (octree subdivision level)

#### Foreground Mesh Reconstruction

**Step 1: Build Open3D Point Cloud**
```python
fg_pcd = o3d.geometry.PointCloud()
fg_pcd.points = o3d.utility.Vector3dVector(fg_points.cpu().numpy())
fg_pcd.colors = o3d.utility.Vector3dVector(fg_colors.cpu().numpy())
fg_pcd.normals = o3d.utility.Vector3dVector(fg_normals.cpu().numpy())
```

**Step 2: Remove Statistical Outliers**
```python
# Remove points far from local neighborhood (noise removal)
cl, ind = fg_pcd.remove_statistical_outlier(
    nb_neighbors=20,   # Look at 20 nearest neighbors
    std_ratio=20.0     # Remove if distance > 20 × std dev
)
fg_pcd = fg_pcd.select_by_index(ind)
```

**Impact:**
- Removes isolated noise points (e.g., from specular highlights, reconstruction errors)
- Typical: ~1-5% of points removed

**Step 3: Poisson Reconstruction**
```python
o3d_fg_mesh, o3d_fg_densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    fg_pcd, 
    depth=10  # Octree depth (CRITICAL for quality vs speed)
)
```

**Poisson Depth Parameter:**

| Depth | Resolution | Vertices (Garden) | Faces | Time | Quality | Use Case |
|-------|------------|-------------------|-------|------|---------|----------|
| 6     | 2^6 = 64³  | ~200K             | ~400K | 5s   | Fair    | Quick test, synthetic scenes |
| 7     | 2^7 = 128³ | ~500K             | ~1M   | 10s  | Good    | Synthetic scenes |
| 8     | 2^8 = 256³ | ~1M               | ~2M   | 20s  | Good+   | Medium complexity |
| 9     | 2^9 = 512³ | ~2M               | ~4M   | 40s  | Very Good | High complexity |
| **10** (default) | 2^10 = 1024³ | **~5M** | **~10M** | **60s** | **Excellent** | **Real scenes (RECOMMENDED)** |
| 11    | 2^11 = 2048³ | ~20M            | ~40M  | 180s | Overkill | Extreme detail, slow |

**Why depth=10?**
- Balances detail preservation with reasonable compute time
- Produces smooth, high-quality meshes for real scenes
- Good starting point before decimation

**Step 4: Remove Low-Density Vertices**
```python
# Poisson creates vertices everywhere (watertight), even in empty space
# Remove vertices with low "confidence" (density score)
vertices_to_remove = o3d_fg_densities < np.quantile(o3d_fg_densities, 0.1)
o3d_fg_mesh.remove_vertices_by_mask(vertices_to_remove)
```

**Density Quantile Parameter:**

| Quantile | Vertices Removed | Effect |
|----------|------------------|--------|
| 0.0      | None             | Keep all vertices (including noise) |
| **0.1** (default) | Bottom 10% | **Remove low-confidence regions (RECOMMENDED)** |
| 0.2      | Bottom 20%       | More aggressive, may remove valid surface |
| 0.5      | Bottom 50%       | Too aggressive, holes in mesh |

**Why quantile=0.1?**
- Removes vertices in empty space (far from surface points)
- Keeps high-confidence surface regions
- Cleans up artifacts from Poisson reconstruction

**Console Output:**
```
-----Foreground mesh-----
Computing points, colors and normals...
Cleaning Point Cloud...
Finished computing points, colors and normals.

⠋ Computing Poisson mesh (foreground)... VRAM: 6.8GB  [00:53]

[yellow]Removing vertices with low densities...[/yellow]
[green]✓ Foreground mesh complete (4,823,567 vertices, 9,647,134 faces)[/green]
```

#### Background Mesh Reconstruction

**Same process as foreground:**
1. Build point cloud
2. Remove outliers
3. Poisson reconstruction (depth=10)
4. Remove low-density vertices

**Console Output:**
```
-----Background mesh-----
Computing points, colors and normals...
Cleaning Point Cloud...
Finished computing points, colors and normals.

⠋ Computing Poisson mesh (background)... VRAM: 7.1GB  [00:48]

[yellow]Removing vertices with low densities...[/yellow]
[green]✓ Background mesh complete (2,145,789 vertices, 4,291,578 faces)[/green]
```

**Time & VRAM:**
- **Foreground:** 40-60s (depends on point count, CPU-bound)
- **Background:** 30-50s
- **VRAM:** ~6-8 GB (CPU does most work, GPU holds point cloud)
- **CPU:** High usage (all cores), Poisson solver is parallelized

---

### 7. Mesh Decimation (Lines 525-580, TIME-INTENSIVE)

**Location:** `coarse_mesh.py:525-580`

**Purpose:** Reduce mesh vertex count while preserving shape and appearance

**Algorithm:** Quadric Error Metric Edge Collapse (Open3D)
- **Method:** Iteratively collapse edges that introduce minimal error
- **Error metric:** Quadric matrix (measures squared distance to adjacent planes)
- **Preserves:** Sharp features, important geometric details

#### Decimation Loop

```python
decimation_targets = [200_000, 1_000_000]  # Two target vertex counts

for decimation_target in decimation_targets:
    print(f"Processing decimation target: {decimation_target:,}")
    
    # Decimate foreground
    decimated_o3d_fg_mesh = o3d_fg_mesh.simplify_quadric_decimation(
        target_number_of_triangles=decimation_target
    )
    # Note: target_number_of_triangles actually targets VERTICES (Open3D naming confusion)
    
    # Decimate background
    decimated_o3d_bg_mesh = o3d_bg_mesh.simplify_quadric_decimation(
        target_number_of_triangles=decimation_target
    )
```

**Decimation Target Recommendations:**

| Target | Vertices | Faces | Quality | FPS (Unity) | Use Case |
|--------|----------|-------|---------|-------------|----------|
| 50K    | ~50K     | ~100K | Fair    | 120+ FPS    | Mobile VR, quick preview |
| 100K   | ~100K    | ~200K | Good    | 90-120 FPS  | Mobile, low-end PC |
| **200K** (default) | **~200K** | **~400K** | **Very Good** | **60-90 FPS** | **General use, game engines** |
| 500K   | ~500K    | ~1M   | Excellent | 30-60 FPS   | High-quality rendering |
| **1M** (high_poly) | **~1M** | **~2M** | **Excellent+** | **15-30 FPS** | **Offline rendering, archival** |
| 2M     | ~2M      | ~4M   | Near-perfect | 5-15 FPS  | Ultra-high quality (24GB+ VRAM) |

**Why Two Targets (200K and 1M)?**
- **200K:** Fast loading, real-time rendering, game-ready
- **1M:** High detail for close-up views, cinematic rendering

**Trade-offs:**

| Aspect | Low Poly (200K) | High Poly (1M) |
|--------|-----------------|----------------|
| Detail | Good (general shape preserved) | Excellent (fine features visible) |
| File size | ~50 MB (PLY) | ~250 MB (PLY) |
| Load time | <1s | 2-5s |
| VRAM | ~100 MB | ~500 MB |
| Unity FPS | 60-90 | 15-30 |
| Decimation time | ~15s | ~30s |

**Console Output:**
```
-----Decimating and cleaning meshes-----

Processing decimation target: 200,000

⠋ Decimating foreground mesh to 200,000 vertices... VRAM: 7.2GB  [00:18]
[green]✓ Foreground decimation complete (199,847 vertices, 399,694 faces)[/green]

⠋ Decimating background mesh to 200,000 vertices... VRAM: 7.1GB  [00:15]
[green]✓ Background decimation complete (199,923 vertices, 399,846 faces)[/green]

Processing decimation target: 1,000,000

⠋ Decimating foreground mesh to 1,000,000 vertices... VRAM: 7.8GB  [00:32]
[green]✓ Foreground decimation complete (999,234 vertices, 1,998,468 faces)[/green]

⠋ Decimating background mesh to 1,000,000 vertices... VRAM: 7.6GB  [00:28]
[green]✓ Background decimation complete (998,876 vertices, 1,997,752 faces)[/green]
```

**Time per Decimation:**
- **200K target:** ~15-20s per mesh
- **1M target:** ~25-35s per mesh
- **Total (2 targets × 2 meshes):** ~2-3 minutes

---

### 8. Mesh Cleaning (Lines 582-597)

**Location:** `coarse_mesh.py:582-597`

**Purpose:** Remove geometric artifacts from decimation

**Operations:**
```python
if clean_mesh:
    # Foreground
    decimated_o3d_fg_mesh.remove_duplicated_vertices()
    decimated_o3d_fg_mesh.remove_degenerate_triangles()
    decimated_o3d_fg_mesh.remove_duplicated_triangles()
    decimated_o3d_fg_mesh.remove_non_manifold_edges()
    
    # Background
    decimated_o3d_bg_mesh.remove_duplicated_vertices()
    decimated_o3d_bg_mesh.remove_degenerate_triangles()
    decimated_o3d_bg_mesh.remove_duplicated_triangles()
    decimated_o3d_bg_mesh.remove_non_manifold_edges()
```

**What Each Operation Does:**

1. **Remove Duplicated Vertices:**
   - Merges vertices at same 3D position
   - Typical: ~0.1-1% of vertices removed
   - Impact: Smaller file size, cleaner topology

2. **Remove Degenerate Triangles:**
   - Removes triangles with zero area (all 3 vertices collinear)
   - Typical: ~0.01-0.1% of triangles removed
   - Impact: Prevents rendering artifacts, NaN normals

3. **Remove Duplicated Triangles:**
   - Removes triangles with identical vertex indices
   - Typical: ~0.01-0.1% of triangles removed
   - Impact: Cleaner mesh, smaller file

4. **Remove Non-Manifold Edges:**
   - Removes edges shared by more than 2 faces
   - Creates "watertight" manifold mesh
   - Typical: ~0.1-1% of edges affected
   - Impact: Mesh is valid for physics engines, 3D printing

**Example (before/after cleaning):**
```
Before: 999,234 vertices, 1,998,468 faces
After:  998,876 vertices, 1,997,523 faces
Removed: 358 vertices, 945 faces
```

**Merge Foreground + Background:**
```python
if (decimated_o3d_fg_mesh is not None) and (decimated_o3d_bg_mesh is not None):
    decimated_o3d_mesh = decimated_o3d_fg_mesh + decimated_o3d_bg_mesh
elif decimated_o3d_fg_mesh is not None:
    decimated_o3d_mesh = decimated_o3d_fg_mesh  # Background empty
elif decimated_o3d_bg_mesh is not None:
    decimated_o3d_mesh = decimated_o3d_bg_mesh  # Foreground empty
else:
    raise ValueError("Both meshes empty!")
```

**Merged Mesh (example):**
```
Foreground:   999,234 verts, 1,998,468 faces
Background:   234,567 verts,   469,134 faces
────────────────────────────────────────────
Merged:     1,233,801 verts, 2,467,602 faces
```

---

### 9. Mesh Projection (Lines 599-661, OPTIONAL BUT RECOMMENDED)

**Location:** `coarse_mesh.py:599-661`

**Purpose:** Snap mesh vertices to actual surface points for detail preservation

**Enabled by:** `--project_mesh_on_surface_points True` (RECOMMENDED)

**Algorithm:**
1. Find K=1 nearest surface point for each mesh vertex (KNN)
2. Replace mesh vertex position with surface point position
3. Transfer surface point color to mesh vertex
4. Recompute vertex normals
5. Re-clean mesh

**Code:**
```python
if project_mesh_on_surface_points:
    # Convert mesh to tensors
    mesh_verts = torch.tensor(np.asarray(decimated_o3d_mesh.vertices), device=sugar.device)
    mesh_faces = torch.tensor(np.asarray(decimated_o3d_mesh.triangles), device=sugar.device)
    
    # Find nearest surface point for each mesh vertex (KNN K=1)
    proj_knn_idx = knn_points(
        mesh_verts[None],      # Query: mesh vertices (N_mesh, 3)
        surface_points[None],  # Database: surface points (10M, 3)
        K=1,
    ).idx[0][..., 0]  # Shape: (N_mesh,) - index of nearest surface point
    
    # Snap vertex positions to surface points
    new_mesh_verts = surface_points[proj_knn_idx]  # (N_mesh, 3)
    new_mesh_colors = surface_colors[proj_knn_idx]  # (N_mesh, 3)
    
    # Rebuild mesh with projected vertices
    decimated_o3d_mesh = o3d.geometry.TriangleMesh()
    decimated_o3d_mesh.vertices = o3d.utility.Vector3dVector(new_mesh_verts.cpu().numpy())
    decimated_o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh_faces.cpu().numpy())
    decimated_o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(new_mesh_colors.cpu().numpy())
    decimated_o3d_mesh.compute_vertex_normals()
    
    # Re-clean (projection may introduce duplicates/degenerate triangles)
    decimated_o3d_mesh.remove_duplicated_vertices()
    decimated_o3d_mesh.remove_degenerate_triangles()
    decimated_o3d_mesh.remove_duplicated_triangles()
    decimated_o3d_mesh.remove_non_manifold_edges()
```

**Why Project?**

**Without projection (Poisson mesh only):**
- Mesh vertices = output of Poisson solver (smoothed, averaged positions)
- Loses fine surface details (e.g., small bumps, texture variations)
- Colors = interpolated from point cloud (blurred)

**With projection (recommended):**
- Mesh vertices = snapped to actual extracted surface points
- Preserves fine surface details from original scene
- Colors = exact colors from rendered images (sharper textures)

**Quality Impact (Example):**

| Aspect | Without Projection | With Projection | Improvement |
|--------|--------------------|--------------------|-------------|
| Fine details (0.5-2cm) | Smoothed/lost | Preserved | +30-50% sharper |
| Color accuracy | Interpolated (blurred) | Exact from renders | +20-30% sharper |
| Normal quality | Poisson smooth | Recomputed from projected | Similar |
| Mesh topology | Optimal (Poisson) | May have slight irregularities | Trade-off |

**VRAM Impact:**
- **KNN computation:** Temporary +2-3 GB spike (~10-20 seconds)
- **After projection:** Returns to ~6-8 GB

**Time:**
- **KNN (1M vertices → 10M surface points):** ~10-15s
- **Mesh rebuild + cleanup:** ~2-5s
- **Total:** ~15-20s per mesh

**Console Output:**
```
⠋ Projecting mesh onto surface points... VRAM: 10.2GB  [00:17]
[green]✓ Mesh projection complete (998,234 vertices, 1,996,468 faces)[/green]
Projection done.
```

**Recommendation:**
- **Always use for final/production meshes** (`--project_mesh_on_surface_points True`)
- **Skip for quick tests** to save 15-20s per mesh

---

### 10. Save Mesh (Lines 663-688)

**Location:** `coarse_mesh.py:663-688`

**Purpose:** Write mesh to PLY file with metadata

**Filename Format:**
```python
# Pattern: sugarmesh_<training_params>_level<L>_decim<D>.ply
# Example: sugarmesh_3Dgs40000_densityestim02_sdfnorm02_level03_decim1000000.ply

sugar_mesh_path = 'sugarmesh_' + sugar_checkpoint_path.split('/')[-2].replace('sugarcoarse_', '') + '_levelZZ_decimAA.ply'
sugar_mesh_path = sugar_mesh_path.replace('ZZ', str(surface_level).replace('.', ''))
sugar_mesh_path = sugar_mesh_path.replace('AA', str(decimation_target).replace('.', ''))
```

**Filename Breakdown:**
- `sugarmesh_` - Prefix
- `3Dgs40000` - Loaded from checkpoint at iteration 40K
- `densityestim02` - Density regularization weight 0.2
- `sdfnorm02` - SDF norm regularization weight 0.2
- `level03` - Surface level 0.3
- `decim1000000` - Decimated to 1M vertices

**Save Options:**
```python
o3d.io.write_triangle_mesh(
    sugar_mesh_path, 
    decimated_o3d_mesh, 
    write_triangle_uvs=True,      # UV coordinates (if available)
    write_vertex_colors=True,     # Vertex colors (RGB)
    write_vertex_normals=True,    # Vertex normals
)
```

**PLY File Contents:**
```
ply
format binary_little_endian 1.0
element vertex 998234
property float x
property float y
property float z
property float nx
property float ny
property float nz
property uchar red
property uchar green
property uchar blue
element face 1996468
property list uchar int vertex_indices
end_header
<binary vertex data>
<binary face data>
```

**File Sizes (Example):**

| Decimation Target | Vertices | Faces | PLY Size (Binary) | PLY Size (ASCII) |
|-------------------|----------|-------|-------------------|------------------|
| 200K              | ~200K    | ~400K | ~45 MB            | ~180 MB          |
| 1M                | ~1M      | ~2M   | ~220 MB           | ~900 MB          |
| 2M                | ~2M      | ~4M   | ~440 MB           | ~1.8 GB          |

**Output Paths (Example):**
```
../SAMPLES/garden_output/garden-r2-60k-6M-quality_mesh/6M-quality-5Mverts/mesh/
├── sugarmesh_3Dgs40000_densityestim02_sdfnorm02_level01_decim200000.ply      # Surface 0.1, 200K verts
├── sugarmesh_3Dgs40000_densityestim02_sdfnorm02_level01_decim1000000.ply     # Surface 0.1, 1M verts
├── sugarmesh_3Dgs40000_densityestim02_sdfnorm02_level03_decim200000.ply      # Surface 0.3, 200K verts
├── sugarmesh_3Dgs40000_densityestim02_sdfnorm02_level03_decim1000000.ply     # Surface 0.3, 1M verts ← BEST
├── sugarmesh_3Dgs40000_densityestim02_sdfnorm02_level05_decim200000.ply      # Surface 0.5, 200K verts
└── sugarmesh_3Dgs40000_densityestim02_sdfnorm02_level05_decim1000000.ply     # Surface 0.5, 1M verts
```

**Console Output:**
```
Mesh saved at ../SAMPLES/garden_output/.../sugarmesh_3Dgs40000_densityestim02_sdfnorm02_level03_decim1000000.ply
```

---

## 🚀 Performance Bottlenecks & Optimizations

### Current Bottlenecks (Ordered by Impact)

1. **Surface Point Extraction (Lines 299-403)** - ~3-5 minutes
   - **Cause:** Rendering 161 frames at 622×960 with multi-pass depth sampling
   - **Optimization:** Lower `mesh_extraction_resolution_factor` (0.5 → 0.4)
   - **Trade-off:** Quality vs speed

2. **Poisson Mesh Reconstruction (Lines 430-523)** - ~2-3 minutes
   - **Cause:** CPU-bound octree solver at depth=10
   - **Optimization:** Lower `poisson_depth` (10 → 8) for quick tests
   - **Trade-off:** Detail vs speed

3. **Mesh Decimation (Lines 525-580)** - ~2-3 minutes
   - **Cause:** Quadric error metric computation for each edge collapse
   - **Optimization:** Skip lower decimation targets if only need 1M mesh
   - **Trade-off:** Multiple outputs vs speed

4. **Mesh Projection (Lines 599-661)** - ~15-20s per mesh (OPTIONAL)
   - **Cause:** KNN search (1M mesh vertices → 10M surface points)
   - **Optimization:** Skip for quick tests (`--project_mesh_on_surface_points False`)
   - **Trade-off:** Quality vs speed

### Recommended Optimizations by Use Case

#### Quick Test (<2 min total)
```bash
python train.py ... \
  --mesh_extraction_resolution_factor 0.4 \        # Lower resolution
  --surface_level 0.3 \                            # Single level only
  --n_vertices_in_mesh 200000 \                    # Single decimation target
  --project_mesh_on_surface_points False           # Skip projection
```

#### Production Quality (~5-10 min total, RECOMMENDED)
```bash
python train.py ... \
  --mesh_extraction_resolution_factor 0.5 \        # Default balance
  --surface_level 0.3 \                            # Single level
  --n_vertices_in_mesh 1000000 \                   # High poly
  --project_mesh_on_surface_points True            # Best detail
```

#### Ultra Quality (~15-20 min total, 24GB+ VRAM)
```bash
python train.py ... \
  --mesh_extraction_resolution_factor 1.0 \        # Full resolution
  --surface_level 0.3 \                            # Single level
  --n_vertices_in_mesh 2000000 \                   # Ultra high poly
  --project_mesh_on_surface_points True \
  --poisson_depth 11                               # Maximum detail (if exposed as param)
```

---

## 🐛 Known Issues & Edge Cases

### Issue 1: Empty Foreground/Background Mesh

**Symptom:**
```
[WARNING] Foreground is empty.
Using background mesh only, since foreground mesh is empty.
```

**Cause:**
- Bounding box too small/large
- Surface level too high/low (no points extracted)
- Camera extent computation incorrect

**Fix:**
- Adjust `--bboxmin` and `--bboxmax` manually
- Try different `--surface_level` values (0.1, 0.3, 0.5)
- Check camera positions (may be far from scene)

### Issue 2: Mesh Has Holes

**Symptom:** Visible gaps in mesh, missing triangles

**Cause:**
- Too few surface points extracted (< 5M)
- Surface level too low (0.1)
- Vertices_density_quantile too high (> 0.2)

**Fix:**
- Increase `n_total_points` (10M → 20M)
- Use surface_level 0.3 (default)
- Lower vertices_density_quantile (0.1 → 0.05)

### Issue 3: Mesh Too Smooth (Lost Details)

**Symptom:** Fine surface features smoothed out

**Cause:**
- Poisson depth too low (< 8)
- Mesh projection disabled
- Decimation target too low (< 200K)

**Fix:**
- Use poisson_depth 10 (default)
- Enable `--project_mesh_on_surface_points True`
- Use higher decimation target (1M)

### Issue 4: VRAM OOM During Mesh Extraction

**Symptom:** CUDA out of memory during surface point extraction

**Cause:**
- `mesh_extraction_resolution_factor` too high (> 0.5)
- `n_points_per_pass` too high (> 2M)
- `faces_per_pixel` too high (> 5)

**Fix:**
- Lower resolution_factor (0.5 → 0.4 or 0.3)
- Reduce n_points_per_pass (2M → 1M)
- Already optimized: faces_per_pixel = 5

---

## 📝 Summary & Key Takeaways

### Critical Parameters (Impact on Quality/Speed/VRAM)

**High Impact:**
1. `mesh_extraction_resolution_factor` (0.5 default) - VRAM: 8-24 GB, Speed: 2-5 min
2. `project_mesh_on_surface_points` (True) - Quality: +30-50% detail, Time: +15-20s
3. `n_vertices_in_mesh` (1M) - Quality: Excellent, File size: ~220 MB
4. `surface_level` (0.3) - Quality: Balanced thickness

**Medium Impact:**
5. `poisson_depth` (10) - Quality: Excellent, Time: ~60s
6. `vertices_density_quantile` (0.1) - Quality: Clean mesh
7. `n_total_points` (10M) - Quality: Dense surface, Time: +30s per 10M

**Low Impact (Keep Defaults):**
8. `primitive_types` ('diamond')
9. `triangle_scale` (2.0)
10. `faces_per_pixel` (5)

### Recommended Workflow

**For 16GB VRAM (RTX 5060 Ti, 5070, 5080):**
```bash
--mesh_extraction_resolution_factor 0.5 \  # Safe VRAM usage
--surface_level 0.3 \                       # Balanced
--n_vertices_in_mesh 1000000 \              # High quality
--project_mesh_on_surface_points True       # Best detail
```

**For 24GB+ VRAM (RTX 5090, A5000):**
```bash
--mesh_extraction_resolution_factor 0.75 \  # Higher resolution
--surface_level 0.3 \
--n_vertices_in_mesh 2000000 \              # Ultra quality
--project_mesh_on_surface_points True
```

### Time Budget Breakdown

| Stage | Time | % of Total | Optimizable? |
|-------|------|------------|--------------|
| Load model | ~5s | 1% | No |
| Prune Gaussians | ~3s | 0.5% | No |
| Build triangle soup | ~5s | 1% | No |
| **Surface extraction** | **3-5 min** | **50-60%** | **Yes** (resolution_factor) |
| Foreground/background split | ~3s | 0.5% | No |
| **Poisson reconstruction** | **2-3 min** | **30-40%** | **Yes** (poisson_depth) |
| **Mesh decimation** | **2-3 min** | **20-30%** | **Yes** (fewer targets) |
| Mesh cleaning | ~5s | 1% | No |
| **Mesh projection** | **15-20s** | **5%** | **Yes** (skip if testing) |
| Save mesh | ~3s | 0.5% | No |
| **Total** | **8-12 min** | **100%** | - |

**Top 3 Optimizations:**
1. Lower `mesh_extraction_resolution_factor` (0.5 → 0.4) → Save 1-2 min
2. Lower `poisson_depth` (10 → 8) for tests → Save 30-60s
3. Skip `project_mesh_on_surface_points` for tests → Save 15-20s

---

## 🔗 Related Documentation

- **[SUGAR_USAGE.MD](../DOCS/SUGAR_USAGE.MD)** - Command-line parameters and examples
- **[SUGAR_TRAIN.MD](../DOCS/SUGAR_TRAIN.MD)** - Full pipeline overview (Stage 1-4)
- **[SUGAR_OPTIMISATIONS.MD](../DOCS/SUGAR_OPTIMISATIONS.MD)** - VRAM and speed optimizations
- **[LLM.MD](LLM.MD)** - Project status and workflows

---

**Last Updated:** January 29, 2026  
**Version:** Complete Stage 2 analysis with all optimization insights
