#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import numpy as np
import open3d as o3d
import cv2
import torch
import random
from random import randint
from datetime import datetime
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import lpips
import sys

# Disable IPython/Jupyter automatic timestamp formatting
if hasattr(sys.stdout, '_new_lines'):
    sys.stdout._new_lines = False
if hasattr(sys.stderr, '_new_lines'):
    sys.stderr._new_lines = False

# Add parent directory to path for console_logger
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import console_logger as log

from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.ply_utils import process_saved_ply
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import gc  # Python garbage collection
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.live import Live

# Get console instance for Progress bars
CONSOLE = log.get_console()

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# Import system monitoring
import psutil
import subprocess

# Try to import pynvml for direct GPU monitoring (faster than nvidia-smi subprocess)
# Don't initialize yet - will init later after CUDA is ready
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    pynvml = None

# RTX 50 Series (Blackwell) Optimizations
torch.backends.cuda.matmul.allow_tf32 = True      # TF32 acceleration for matmul
torch.backends.cudnn.allow_tf32 = True            # TF32 acceleration for cuDNN
torch.backends.cudnn.benchmark = True             # Auto-tune for optimal algorithms
# Note: expandable_segments disabled due to free() pointer corruption with custom CUDA modules
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Global variable for tracking Gaussian count changes
prev_gaussian_count = None

def get_system_resources():
    """Get current VRAM and RAM usage with percentages."""
    resources = {}
    
    # VRAM monitoring - show total memory pressure (can exceed 100% when using shared memory)
    if torch.cuda.is_available():
        # Get reserved memory (includes both dedicated + shared if spilling over)
        vram_reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
        
        # Get dedicated VRAM capacity
        mem_free, mem_total = torch.cuda.mem_get_info()
        vram_dedicated = mem_total / (1024**3)  # GB (16GB for your card)
        
        # Calculate percentage: >100% means using shared memory (swap hell)
        vram_percent = (vram_reserved / vram_dedicated) * 100
        
        resources['vram_gb'] = vram_reserved
        resources['vram_total_gb'] = vram_dedicated
        resources['vram_percent'] = vram_percent
    else:
        resources['vram_gb'] = 0
        resources['vram_total_gb'] = 0
        resources['vram_percent'] = 0
    
    # RAM monitoring
    ram_info = psutil.virtual_memory()
    resources['ram_gb'] = ram_info.used / (1024**3)
    resources['ram_total_gb'] = ram_info.total / (1024**3)
    resources['ram_percent'] = ram_info.percent
    
    return resources

def print_memory_mode(args):
    """Print memory loading strategy."""
    if getattr(args, 'low_dram', False):
        cache_gb = getattr(args, 'image_cache_gb', 2.0)
        log.log(f"✓ [green]Lazy loading enabled[/green] (LRU cache limit: {cache_gb}GB)")
        log.log(f"  [dim]→ Loads images on-demand, auto-calculates cache size based on resolution[/dim]")
        log.log(f"  [dim]→ Adjust with --image_cache_gb (default: 2.0GB)[/dim]")
        log.log("  [dim]→ Remove --low_dram flag for eager loading (all images at startup)[/dim]")
    else:
        log.log("⚡ [yellow]Eager loading enabled[/yellow] (all images loaded at startup)")
        log.log("  [dim]→ Faster training, loads all images into RAM immediately[/dim]")
        log.log("  [dim]→ Use --low_dram --image_cache_gb <GB> to limit RAM usage[/dim]")

@torch.no_grad()
def create_offset_gt(image, offset):
    height, width = image.shape[1:]
    meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    id_coords = torch.from_numpy(id_coords).cuda()
    
    id_coords = id_coords.permute(1, 2, 0) + offset
    id_coords[..., 0] /= (width - 1)
    id_coords[..., 1] /= (height - 1)
    id_coords = id_coords * 2 - 1
    
    image = torch.nn.functional.grid_sample(image[None], id_coords[None], align_corners=True, padding_mode="border")[0]
    return image

def log_config_text(tb_writer, dataset, opt, scene):
    """Log configuration tables to TensorBoard TEXT tab."""
    if not tb_writer:
        return
    
    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
    
    # Get camera info from first training camera
    train_cams = scene.getTrainCameras()
    first_cam = train_cams[0] if train_cams else None
    
    # Extract COLMAP camera model and 3D point info
    camera_model = "N/A"
    focal_length = "N/A"
    image_size = "N/A"
    training_resolution = "N/A"
    resolution_note = ""
    sparse_points = "N/A"
    
    if first_cam:
        # Image dimensions (original, before downsampling)
        orig_width = first_cam.image_width
        orig_height = first_cam.image_height
        image_size = f"{orig_width} x {orig_height} px"
        
        # Calculate actual training resolution after -r downsampling
        if dataset.resolution > 1:
            train_width = orig_width // dataset.resolution
            train_height = orig_height // dataset.resolution
            training_resolution = f"{train_width} x {train_height} px"
            resolution_note = f" (downsampled by -r {dataset.resolution})"
        else:
            training_resolution = image_size
            resolution_note = " (full resolution)"
        
        # Focal length (convert from FoV back to pixels for first camera)
        from utils.graphics_utils import fov2focal
        focal_x = fov2focal(first_cam.FoVx, orig_width)
        focal_y = fov2focal(first_cam.FoVy, orig_height)
        if abs(focal_x - focal_y) < 1.0:
            focal_length = f"{focal_x:.1f} px (SIMPLE_PINHOLE)"
            camera_model = "SIMPLE_PINHOLE"
        else:
            focal_length = f"fx={focal_x:.1f}, fy={focal_y:.1f} px (PINHOLE)"
            camera_model = "PINHOLE"
    
    # Try to get 3D point count from initial point cloud
    try:
        ply_path = os.path.join(dataset.source_path, "sparse/0/points3D.ply")
        if os.path.exists(ply_path):
            from plyfile import PlyData
            plydata = PlyData.read(ply_path)
            sparse_points = f"{len(plydata['vertex']):,}"
    except:
        pass
    
    # Config/01_Dataset
    dataset_info = f"""
## 📊 Dataset & Camera Information

| Property | Value |
|----------|-------|
| **Scene Name** | {os.path.basename(dataset.source_path)} |
| **Training Views** | {len(scene.getTrainCameras())} images |
| **Test Views** | {len(scene.getTestCameras())} images |
| **Image Size (Original)** | {image_size} |
| **Training Resolution** | **{training_resolution}**{resolution_note} |
| **Resolution Divisor (-r)** | {dataset.resolution} |
| **Camera Model** | {camera_model} (COLMAP reconstruction) |
| **Focal Length** | {focal_length} |
| **COLMAP Sparse Points** | {sparse_points} (initial 3D reconstruction) |
| **Background Color** | {'White (1.0)' if dataset.white_background else 'Black (0.0)'} |
| **SH Degree** | {dataset.sh_degree} (max spherical harmonics order for view-dependent color) |
| **Kernel Size** | {dataset.kernel_size} (anti-aliasing filter size) |
| **Ray Jitter** | {'✅ Enabled (reduces aliasing)' if dataset.ray_jitter else '❌ Disabled'} |
| **Resample GT** | {'✅ Enabled (GT images resampled to match render resolution)' if dataset.resample_gt_image else '❌ Disabled'} |

⚠️ **PSNR Comparison Note:** Only compare PSNR between models trained at the SAME resolution!  
Lower resolution (-r 8) will show higher PSNR but worse quality. Higher resolution (-r 1) will show lower PSNR but best quality.
"""
    tb_writer.add_text('Config/01_Dataset', dataset_info, 0)
    
    # Config/02_Optimization
    opt_info = f"""
## ⚙️ Optimization Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Total Iterations** | {opt.iterations:,} | Training steps (full training duration) |
| **Position LR (init→final)** | {opt.position_lr_init:.6f} → {opt.position_lr_final:.10f} | Gaussian center positions (decays over training) |
| **Feature LR** | {opt.feature_lr:.6f} | SH color coefficients (view-dependent appearance) |
| **Opacity LR** | {opt.opacity_lr:.6f} | Transparency values (0=transparent, 1=opaque) |
| **Scaling LR** | {opt.scaling_lr:.6f} | Gaussian size/radius in 3D space |
| **Rotation LR** | {opt.rotation_lr:.6f} | Gaussian orientation (quaternion) |
| **λ DSSIM** | {opt.lambda_dssim:.4f} | Structural similarity weight (0.2 = 20% SSIM, 80% L1) |
| **Densify Grad Threshold** | {opt.densify_grad_threshold:.6f} | View-space gradient threshold to split/clone gaussians |
| **Densification Interval** | {opt.densification_interval} iters | Check every N iterations for under-reconstructed areas |
| **Opacity Reset Interval** | {opt.opacity_reset_interval:,} iters | Reset low-opacity gaussians to prevent floaters |
| **Densify From** | {opt.densify_from_iter:,} iters | Start adding gaussians (initial warmup period) |
| **Densify Until** | {opt.densify_until_iter:,} iters | Stop adding gaussians (refinement-only after this) |
| **Percent Dense** | {opt.percent_dense:.4f} | Spatial extent threshold for densification culling |
"""
    tb_writer.add_text('Config/02_Optimization', opt_info, 0)
    
    # Config/03_Performance
    memory_mode = "Eager Loading (all in RAM)" if getattr(dataset, 'high_dram', False) else f"Lazy Loading (LRU cache: {getattr(dataset, 'image_cache_size', 20)} images)"
    data_device = dataset.data_device if hasattr(dataset, 'data_device') else "cuda"
    
    # Get PCIe info
    pcie_info = "N/A"
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=pcie.link.gen.current,pcie.link.width.current', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=1)
        if result.returncode == 0:
            values = result.stdout.strip().split(',')
            if len(values) >= 2:
                pcie_gen = values[0].strip()
                pcie_width = values[1].strip()
                pcie_info = f"Gen {pcie_gen} x{pcie_width}"
    except:
        pass
    
    perf_info = f"""
## 💾 Memory & Performance

| Setting | Value |
|---------|-______|
| **GPU Model** | {gpu_name} |
| **VRAM** | {gpu_memory_gb:.1f} GB |
| **PCIe Bus** | {pcie_info} |
| **TF32 Acceleration** | ✅ Enabled |
| **cuDNN Benchmark** | ✅ Enabled |
| **Memory Mode** | {memory_mode} |
| **Data Device** | {data_device.upper()} |
| **3D Filter Updates** | Every 100 iters (after densification) |
| **Kernel Size** | {dataset.kernel_size} |
"""
    tb_writer.add_text('Config/03_Performance', perf_info, 0)
    
    # Log exact training command for reproducibility
    import sys
    training_command = ' '.join(sys.argv)
    command_info = f"""
# Exact Training Command (Copy & Paste to Reproduce)

```bash
python {training_command}
```

**Working Directory:** `{os.getcwd()}`

**To rerun this exact training:**
1. Navigate to: `cd {os.getcwd()}`
2. Copy the command above
3. Paste and run in terminal

**To resume from checkpoint:**
```bash
python {training_command.replace(f'--experiment_name "{dataset.experiment_name}"', f'--start_checkpoint ./output/{dataset.experiment_name or "<timestamp>"}/chkpntXXXX.pth --experiment_name "{dataset.experiment_name or "<timestamp>"}-resumed"' if dataset.experiment_name else '--start_checkpoint ./output/<timestamp>/chkpntXXXX.pth')}
```

**Note:** Replace `chkpntXXXX.pth` with actual checkpoint file (e.g., `chkpnt7000.pth`)
"""
    tb_writer.add_text('Config/00_Training_Command', command_info, 0)

# Global variable to store last test metrics for PLY analysis
last_test_metrics = {'psnr': None, 'ssim': None, 'lpips': None}

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    global prev_gaussian_count
    global last_test_metrics
    
    first_iter = 0
    print_memory_mode(dataset)
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    if not dataset.eval_only:  # Skip optimizer setup for eval-only mode (saves 2-3GB VRAM)
        gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)
    
    # Log configuration tables to TensorBoard TEXT tab
    log_config_text(tb_writer, dataset, opt, scene)
    
    # Initialize Gaussian count tracking
    prev_gaussian_count = gaussians.get_xyz.shape[0]

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    trainCameras = scene.getTrainCameras().copy()
    testCameras = scene.getTestCameras().copy()
    allCameras = trainCameras + testCameras
    
    # Display COLMAP scene quality info in table
    scene_table = Table(title="[bold cyan]COLMAP Scene Info[/bold cyan]", 
                       show_header=True, header_style="bold cyan", border_style="cyan", width=80)
    scene_table.add_column("Property", style="cyan", justify="left", width=30)
    scene_table.add_column("Value", style="yellow bold", justify="right", width=20)
    scene_table.add_column("Notes", style="dim", justify="left", width=30)
    
    num_points = gaussians.get_xyz.shape[0]
    scene_table.add_row("Initial 3D Points", f"{num_points:,}", "From COLMAP reconstruction")
    scene_table.add_row("Training Cameras", f"{len(trainCameras)}", "Used for training Gaussians")
    scene_table.add_row("Test Cameras", f"{len(testCameras)}", "Used for validation/metrics")
    scene_table.add_row("Scene Extent", f"{scene.cameras_extent:.2f} units", "Bounding sphere radius")
    
    if trainCameras:
        # Calculate camera intrinsics
        avg_fovx = sum(cam.FoVx for cam in trainCameras) / len(trainCameras)
        avg_fovy = sum(cam.FoVy for cam in trainCameras) / len(trainCameras)
        
        # Calculate focal length in pixels (fx = width / (2 * tan(fov_x / 2)))
        focal_lengths_x = [cam.image_width / (2 * np.tan(cam.FoVx / 2)) for cam in trainCameras]
        focal_lengths_y = [cam.image_height / (2 * np.tan(cam.FoVy / 2)) for cam in trainCameras]
        avg_fx = sum(focal_lengths_x) / len(focal_lengths_x)
        avg_fy = sum(focal_lengths_y) / len(focal_lengths_y)
        
        # Image dimensions (check if they vary)
        widths = set(cam.image_width for cam in trainCameras)
        heights = set(cam.image_height for cam in trainCameras)
        
        if len(widths) == 1 and len(heights) == 1:
            img_dims = f"{list(widths)[0]} × {list(heights)[0]} px"
        else:
            img_dims = f"{min(widths)}-{max(widths)} × {min(heights)}-{max(heights)} px"
        
        scene_table.add_row("Image Dimensions", img_dims, "Camera sensor resolution")
        scene_table.add_row("Average FoV", f"{np.rad2deg(avg_fovx):.1f}° × {np.rad2deg(avg_fovy):.1f}°", "Horizontal × Vertical")
        scene_table.add_row("Focal Length (avg)", f"{avg_fx:.1f} × {avg_fy:.1f} px", "fx × fy in pixels")
    
    log.log()
    log.print_table(scene_table)
    
    # Clear VRAM cache before training starts
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        log.log("[green]✓ VRAM cache cleared[/green]")
    
    # highresolution index
    highresolution_index = []
    for index, camera in enumerate(trainCameras):
        if camera.image_width >= 800:
            highresolution_index.append(index)

    # Display resolution info
    first_cam = trainCameras[0] if trainCameras else None
    if first_cam:
        orig_w, orig_h = first_cam.image_width, first_cam.image_height
        if dataset.resolution > 1:
            train_w, train_h = orig_w // dataset.resolution, orig_h // dataset.resolution
            log.log(f"[cyan]Image Resolution:[/cyan]")
            log.log(f"  Original: {orig_w} × {orig_h} px")
            log.log(f"  Training: [bold green]{train_w} × {train_h} px[/bold green] (downsampled by [yellow]-r {dataset.resolution}[/yellow])")
            log.log(f"  [dim]Note: PSNR only comparable at SAME training resolution[/dim]")
        else:
            log.log(f"[cyan]Image Resolution:[/cyan] [bold green]{orig_w} × {orig_h} px[/bold green] (full resolution)")
    
    # Compute 3D filter with summary display
    log.log(f"[cyan]Computing 3D filter for {len(trainCameras)} cameras[/cyan] (batched 8 cameras/batch)...")
    gaussians.compute_3D_filter(cameras=trainCameras)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    
    # Use Rich Progress bar with resource monitoring
    log.log(f"[green]Starting training...[/green]")
    
    # Flush unallocated memory before monitoring baseline
    log.log(f"[dim]Flushing unallocated memory...[/dim]")
    gc.collect()
    
    # Initialize pynvml AFTER CUDA is initialized and ready
    nvml_handle = None
    if PYNVML_AVAILABLE:
        try:
            pynvml.nvmlInit()
            nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            log.log(f"[dim]✓ pynvml GPU monitoring initialized[/dim]")
        except Exception as e:
            log.log(f"[yellow]⚠ pynvml init failed, falling back to nvidia-smi: {e}[/yellow]")
            nvml_handle = None
    
    # Test resource monitoring before progress bar
    try:
        test_resources = get_system_resources()
        log.log(f"[dim]Initial VRAM: {test_resources['vram_percent']:.0f}%/{test_resources['vram_gb']:.1f}GB, RAM: {test_resources['ram_percent']:.0f}%/{test_resources['ram_gb']:.1f}GB[/dim]")
    except Exception as e:
        log.log(f"[red]Warning: Resource monitoring failed: {e}[/red]")
    
    # LPIPS status message
    log.log("")
    if opt.enable_lpips:
        log.log(f"[green]✓ LPIPS metric enabled[/green] - Adds ~1min per evaluation for perceptual quality assessment")
        log.log(f"[dim]  Benefits: Standard metric in 3D reconstruction papers, detects perceptual artifacts[/dim]")
        log.log(f"[dim]  To disable: Add --enable_lpips False to command line[/dim]")
    
    # Train/test split status (only show if --eval is enabled)
    if dataset.model_path != "" and args.eval:
        train_cam_count = len(scene.getTrainCameras())
        test_cam_count = len(scene.getTestCameras())
        total_cams = train_cam_count + test_cam_count
        log.log(f"[green]✓ Train/test split[/green]: {train_cam_count} train / {test_cam_count} test cameras (total {total_cams})")
        log.log(f"[dim]  Test cameras evenly distributed (includes first & last frame) for {args.test_camera_count} requested[/dim]")
        log.log(f"[dim]  Benefits: More training data = better model, fewer test cameras = less VRAM during evaluation[/dim]")
    
    # Initialize LPIPS model once at startup if enabled
    if opt.enable_lpips:
        log.log("[cyan]⏳ Initializing LPIPS model (downloading VGG weights on first run, ~500MB)...[/cyan]")
        import contextlib
        import io
        import warnings
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lpips_fn = lpips.LPIPS(net='vgg').cuda()
        log.log("[green]✓ LPIPS model ready[/green]")
    else:
        log.log(f"[yellow]⚠ LPIPS metric disabled[/yellow] - Faster evaluation but missing perceptual quality metric")
        lpips_fn = None
    
    # Eval-only mode status
    if dataset.eval_only:
        log.log(f"[green]✓ Eval-only mode[/green] - Optimizer skipped (saves 2-3GB VRAM)")
        log.log(f"[dim]  Requires: --start_checkpoint <path> to load trained model[/dim]")
    log.log("")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green", finished_style="green bold", bar_width=10),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[cyan]{task.completed}/{task.total}[/cyan]"),
        TextColumn("L:[magenta]{task.fields[loss]:.4f}[/magenta]"),
        TextColumn("{task.fields[speed]}"),
        TextColumn("V:[cyan]{task.fields[vram]}[/cyan]"),
        TextColumn("R:[yellow]{task.fields[ram]}[/yellow]"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=CONSOLE,
        refresh_per_second=2,  # Lower refresh rate to reduce rendering artifacts
        expand=False,  # Don't expand to full terminal width
        transient=False,  # Keep progress bar visible (don't clear)
        auto_refresh=True,  # Let Rich handle refresh automatically
    ) as progress:
        task = progress.add_task(
            "Training",
            total=opt.iterations - first_iter + 1,
            loss=0.0,
            speed="--",
            vram="--",
            ram="--"
        )
        
        # Show initial state with resources
        try:
            resources = get_system_resources()
            vram_str = f"{resources['vram_percent']:.0f}%/{resources['vram_gb']:.1f}GB"
            ram_str = f"{resources['ram_percent']:.0f}%/{resources['ram_gb']:.1f}GB"
            progress.update(task, completed=0, loss=0.0, vram=vram_str, ram=ram_str)
        except Exception as e:
            log.log(f"[red]Progress update failed: {e}[/red]")
            progress.update(task, completed=0, loss=0.0, vram="--", ram="--")
        
        first_iter += 1
        
        try:
            for iteration in range(first_iter, opt.iterations + 1):        
                if network_gui.conn == None:
                    network_gui.try_connect()
                while network_gui.conn != None:
                    try:
                        net_image_bytes = None
                        custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                        if custom_cam != None:
                            net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                            net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                        network_gui.send(net_image_bytes, dataset.source_path)
                        if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                            break
                    except Exception as e:
                        network_gui.conn = None

            iter_start.record()

            gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            
            # Pick a random high resolution camera
            if random.random() < 0.3 and dataset.sample_more_highres:
                viewpoint_cam = trainCameras[highresolution_index[randint(0, len(highresolution_index)-1)]]
                
            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            #TODO ignore border pixels
            if dataset.ray_jitter:
                subpixel_offset = torch.rand((int(viewpoint_cam.image_height), int(viewpoint_cam.image_width), 2), dtype=torch.float32, device="cuda") - 0.5
                # subpixel_offset *= 0.0
            else:
                subpixel_offset = None
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size=dataset.kernel_size, subpixel_offset=subpixel_offset)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # Loss
            # Phase 2 optimization: Keep GT image on CPU, transfer to GPU only during loss computation
            gt_image_cpu = viewpoint_cam.original_image.cpu() if viewpoint_cam.original_image.is_cuda else viewpoint_cam.original_image
            # sample gt_image with subpixel offset
            if dataset.resample_gt_image:
                gt_image_cpu = create_offset_gt(gt_image_cpu.cuda(), subpixel_offset).cpu()
            
            # Transfer to GPU only for loss computation, then immediately release
            gt_image = gt_image_cpu.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            del gt_image  # Free GPU memory immediately after loss computation
            loss.backward()

            iter_end.record()

            with torch.no_grad():
                # Update progress bar with loss and resources (every 10 iterations for smooth display)
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                
                # Update progress bar every 10 iterations (smooth UI updates)
                if iteration % 10 == 0:
                    # Get system resources (lightweight ~1ms)
                    resources = get_system_resources()
                    vram_str = f"{resources['vram_percent']:.0f}%/{resources['vram_gb']:.1f}GB"
                    ram_str = f"{resources['ram_percent']:.0f}%/{resources['ram_gb']:.1f}GB"
                    
                    # Calculate iteration speed
                    elapsed_ms = iter_start.elapsed_time(iter_end)
                    speed_str = f"{1000.0/elapsed_ms:.1f}it/s" if elapsed_ms > 0 else "--"
                    
                    progress.update(task, advance=10, loss=ema_loss_for_log, vram=vram_str, ram=ram_str, speed=speed_str)

            # Log to TensorBoard every 50 iterations (reduced overhead vs every 10)
            enhanced_training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, dataset.kernel_size), gaussians, opt, dataset, nvml_handle, lpips_fn, log_metrics=(iteration % 50 == 0))
            
            # Track Gaussian count changes (after densification)
            if tb_writer and iteration > first_iter:
                current_count = gaussians.get_xyz.shape[0]
                count_change = current_count - prev_gaussian_count
                
                if count_change != 0:
                    tb_writer.add_scalar('gaussians/net_change_K', count_change / 1000.0, iteration)
                
                prev_gaussian_count = current_count
            
            # Progress table (every 5000 iterations)
            if iteration % 5000 == 0 and iteration > 0:
                elapsed_min = (iter_start.elapsed_time(iter_end) * iteration) / 60000.0
                remaining_iters = opt.iterations - iteration
                eta_min = (iter_start.elapsed_time(iter_end) * remaining_iters) / 60000.0
                fps = 1000.0 / iter_start.elapsed_time(iter_end) if iter_start.elapsed_time(iter_end) > 0 else 0
                gpu_gb = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
                
                # Create progress table
                progress_table = Table(title=f"[bold cyan]Training Progress - Iteration {iteration:,}[/bold cyan]", 
                                     show_header=True, header_style="bold magenta", border_style="cyan")
                progress_table.add_column("Metric", style="cyan", justify="left")
                progress_table.add_column("Value", style="yellow", justify="right")
                progress_table.add_column("Status", style="green", justify="center")
                
                progress_pct = 100.0 * iteration / opt.iterations
                progress_table.add_row("Progress", f"{iteration:,} / {opt.iterations:,}", f"{progress_pct:.1f}%")
                progress_table.add_row("Time Elapsed", f"{elapsed_min:.1f} min", f"ETA: {eta_min:.1f} min")
                progress_table.add_row("Loss (EMA)", f"{ema_loss_for_log:.4f}", "")
                progress_table.add_row("Gaussians", f"{gaussians.get_xyz.shape[0]:,}", f"{gaussians.get_xyz.shape[0]/1000.0:.1f}K")
                progress_table.add_row("GPU Memory", f"{gpu_gb:.2f} GB", "")
                progress_table.add_row("Speed", f"{fps:.2f} it/s", "")
                
                log.print_table(progress_table)
            
            if (iteration in saving_iterations):
                progress.console.print(f"[green bold]Iter {iteration}: Saving Gaussians[/green bold]")
                scene.save(iteration)
                
                # Analyze and cleanup saved PLY
                ply_path = os.path.join(scene.model_path, f"point_cloud/iteration_{iteration}/point_cloud.ply")
                process_saved_ply(
                    ply_path,
                    iteration=iteration,
                    loss=ema_loss_for_log,
                    test_metrics=last_test_metrics,
                    console=progress.console
                )

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # Track VRAM before pruning (use nvidia-smi for consistency with progress bar)
                    if torch.cuda.is_available():
                        resources_before = get_system_resources()
                        vram_before = resources_before['vram_gb'] * 1024  # Convert GB to MB
                        gaussians_before = gaussians.get_xyz.shape[0]
                    
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    cloned, split, pruned = gaussians.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity_threshold, scene.cameras_extent, size_threshold)
                    
                    # Log densification stats to TensorBoard
                    if tb_writer:
                        tb_writer.add_scalar('gaussians/added_K', (cloned + split) / 1000.0, iteration)
                        tb_writer.add_scalar('gaussians/removed_K', pruned / 1000.0, iteration)
                        tb_writer.add_scalar('gaussians/net_change_K', (cloned + split - pruned) / 1000.0, iteration)
                    
                    # Binary search hard cap: Enforce absolute Gaussian count limit
                    if opt.target_gaussian_count > 0:
                        current_count = gaussians.get_xyz.shape[0]
                        if current_count > opt.target_gaussian_count:
                            # Find exact threshold to hit target count
                            threshold = gaussians.find_pruning_threshold(
                                opt.target_gaussian_count,
                                search_min=opt.binary_search_min,
                                search_max=opt.binary_search_max,
                                max_iterations=opt.binary_search_iterations
                            )
                            prune_mask = (gaussians.get_opacity < threshold).squeeze()
                            gaussians.prune_points(prune_mask)
                            
                            progress.console.print(
                                f"[bold red]🎯 Hard cap enforced:[/bold red] "
                                f"{current_count:,} → {gaussians.get_xyz.shape[0]:,} Gaussians "
                                f"(threshold={threshold:.4f})"
                            )
                    
                    # Log VRAM change after pruning (use nvidia-smi for consistency)
                    if torch.cuda.is_available():
                        resources_after = get_system_resources()
                        vram_after = resources_after['vram_gb'] * 1024  # Convert GB to MB
                        gaussians_after = gaussians.get_xyz.shape[0]
                        vram_change = vram_after - vram_before
                        gaussian_change = gaussians_after - gaussians_before
                        
                        progress.console.print(
                            f"[yellow]Iter {iteration} Densify+Prune:[/yellow] "
                            f"Gaussians: [cyan]{gaussians_after/1000:.0f}K ({gaussian_change/1000:+.0f}K)[/cyan] | "
                            f"VRAM: {vram_after:.0f} MB ({vram_change:+.0f} MB)"
                        )
                    
                    progress.stop()  # Pause progress bar during 3D filter computation
                    gaussians.compute_3D_filter(cameras=trainCameras)
                    progress.start()  # Resume progress bar

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    progress.stop()  # Pause progress bar
                    gaussians.compute_3D_filter(cameras=trainCameras)
                    progress.start()  # Resume progress bar

            if iteration % 500 == 0 and iteration > opt.densify_until_iter:
                if iteration < opt.iterations - 500:
                    # don't update in the end of training
                    progress.stop()
                    progress.console.print(f"[cyan]Iter {iteration}: Updating 3D filter[/cyan]")
                    gaussians.compute_3D_filter(cameras=trainCameras)
                    progress.start()
        
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                
            # Periodic memory cleanup every 500 iterations
            if iteration % 500 == 0:
                gc.collect()
            
            # Memory cleanup - prevent VRAM creep (gt_image already deleted after loss computation in Phase 2)
            del render_pkg, image, viewspace_point_tensor, visibility_filter, radii
            del Ll1, loss, gt_image_cpu
            if subpixel_offset is not None:
                del subpixel_offset
            if iteration % 500 == 0:
                torch.cuda.empty_cache()

                if (iteration in checkpoint_iterations):
                    checkpoint_path = scene.model_path + "/chkpnt" + str(iteration) + ".pth"
                    progress.console.print(f"[blue bold]Iter {iteration}: Saving Checkpoint[/blue bold]")
                    progress.console.print(f"  → {checkpoint_path}")
                    torch.save((gaussians.capture(), iteration), checkpoint_path)
        
        except KeyboardInterrupt:
            # Graceful exit on CTRL-C
            progress.stop()
            progress.console.print(f"\n\n[bold yellow]⚠️  Training interrupted by user (CTRL-C)[/bold yellow]")
            progress.console.print(f"[cyan]Progress: Iteration {iteration:,} / {opt.iterations:,} ({100.0 * iteration / opt.iterations:.1f}%)[/cyan]")
            
            # Save emergency checkpoint
            emergency_checkpoint = scene.model_path + f"/chkpnt{iteration}_interrupted.pth"
            progress.console.print(f"\n[bold cyan]💾 Saving emergency checkpoint...[/bold cyan]")
            torch.save((gaussians.capture(), iteration), emergency_checkpoint)
            progress.console.print(f"[green]✓ Checkpoint saved: {emergency_checkpoint}[/green]")
            
            # Show resume command
            progress.console.print(f"\n[bold]To resume training from iteration {iteration}:[/bold]")
            progress.console.print(f"[dim]  --start_checkpoint {emergency_checkpoint}[/dim]\n")
            
            # Cleanup
            if nvml_handle is not None:
                try:
                    pynvml.nvmlShutdown()
                except:
                    pass
            
            if tb_writer:
                tb_writer.close()
            
            return  # Exit gracefully
    
    # Save final iteration if not already in save_iterations
    final_iteration = opt.iterations
    if final_iteration not in saving_iterations:
        progress.console.print(f"\n[green bold]Saving final iteration {final_iteration}[/green bold]")
        scene.save(final_iteration)
        
        # Analyze and cleanup final PLY (output written to ply_analysis.txt)
        ply_path = os.path.join(scene.model_path, f"point_cloud/iteration_{final_iteration}/point_cloud.ply")
        log.log(f"[dim]→ Running PLY analysis and cleanup...[/dim]")
        process_saved_ply(
            ply_path,
            iteration=final_iteration,
            loss=ema_loss_for_log,
            test_metrics=last_test_metrics,
            console=None  # Don't show verbose output in console
        )
    
    # Cleanup pynvml handle at end of training
    if nvml_handle is not None:
        try:
            pynvml.nvmlShutdown()
        except:
            pass

def prepare_output_and_logger(args):    
    if not args.model_path:
        # Default: create output folder next to input dataset
        # e.g., /path/to/garden -> /path/to/garden_output/experiment_name/
        dataset_path = args.source_path.rstrip('/')
        output_base = f"{dataset_path}_output"
        
        # Use experiment_name if provided, otherwise use timestamp
        if hasattr(args, 'experiment_name') and args.experiment_name:
            args.model_path = os.path.join(output_base, args.experiment_name)
        else:
            # No experiment name - use timestamp to create unique folder
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.model_path = os.path.join(output_base, timestamp_str)
    
    # Check if output folder already exists
    if os.path.exists(args.model_path):
        import shutil
        log.log(f"[yellow bold]⚠️  Output folder already exists:[/yellow bold]")
        log.log(f"[yellow]   {args.model_path}[/yellow]")
        
        # Check if it has content
        dir_contents = os.listdir(args.model_path)
        if dir_contents:
            log.log(f"[yellow]Folder contains {len(dir_contents)} items[/yellow]")
        
        if args.delete_first:
            log.log(f"[cyan]--delete_first enabled: Auto-deleting existing folder...[/cyan]")
            shutil.rmtree(args.model_path)
            log.log(f"[green]✓ Folder deleted[/green]")
        else:
            response = input("Delete existing folder and continue? [y/N]: ").strip().lower()
            
            if response == 'y' or response == 'yes':
                log.log(f"[red]Deleting existing folder...[/red]")
                shutil.rmtree(args.model_path)
                log.log(f"[green]✓ Folder deleted[/green]")
            else:
                log.log(f"[red]Training cancelled by user.[/red]")
                log.log(f"[dim]Tip: Use --experiment_name to create a unique output folder, or use --delete_first to auto-delete[/dim]")
                sys.exit(0)
        
    # Set up output folder
    log.log(f"[bold]Output folder:[/bold] {args.model_path}")
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
        
        # Determine TensorBoard logdir (parent for comparison, or specific run)
        parent_dir = os.path.dirname(args.model_path)
        if parent_dir and os.path.basename(parent_dir).endswith('_output'):
            # Using automatic naming - show parent directory for run comparison
            tb_logdir = parent_dir
            comparison_note = " (compare multiple runs)"
        else:
            # Using manual path - show specific directory
            tb_logdir = args.model_path
            comparison_note = ""
        
        log.log("[cyan]TensorBoard:[/cyan] Please open in another terminal:")
        log.log(f"[dim]  tensorboard --logdir \"{tb_logdir}\" --port 6006 --bind_all[/dim]")
        log.log(f"[dim]  Then open: http://localhost:6006[/dim]")
        if comparison_note:
            log.log(f"[dim]  Note: Logging to parent directory{comparison_note}[/dim]")
    else:
        log.log("[yellow]Tensorboard not available: not logging progress[/yellow]")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, opt, dataset, gaussians=None, lpips_fn=None, log_metrics=True):
    global last_test_metrics
    
    if tb_writer and log_metrics:
        tb_writer.add_scalar('1_train/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('1_train/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('1_train/iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        log.log(f"[cyan]📊 Starting evaluation at iteration {iteration}...[/cyan]")
        torch.cuda.empty_cache()
        
        # LPIPS model already initialized at startup if enabled
        # NOTE: torch.compile disabled - Triton PTX codegen doesn't support sm_120 (Blackwell RTX 5000 series)
        # Would provide 20-40% speedup on older GPUs (sm_86/89: RTX 3000/4000 series)
        # Error: "ptxas fatal: Value 'sm_120' is not defined for option 'gpu-name'"
        
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                log.log(f"[dim]  → Evaluating {config['name']} set ({len(config['cameras'])} cameras)...[/dim]")
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0 if opt.enable_lpips else None
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    
                    # Phase 2 optimization: Cache GT image on CPU instead of CUDA to save ~2GB VRAM
                    if not hasattr(viewpoint, '_cached_gt_cpu'):
                        viewpoint._cached_gt_cpu = torch.clamp(viewpoint.original_image.cpu(), 0.0, 1.0)
                    # Transfer to GPU only when needed for metric computation
                    gt_image = viewpoint._cached_gt_cpu.cuda()
                    
                    if tb_writer and (idx < 5):
                        # Phase 2: Move to CPU IMMEDIATELY to prevent GPU memory retention
                        image_cpu = image.detach().cpu()
                        gt_image_cpu = gt_image.detach().cpu()
                        
                        # 1. RGB Render
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), 
                                           image_cpu[None], global_step=iteration)
                        
                        # 2. Ground Truth (only first test iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), 
                                               gt_image_cpu[None], global_step=iteration)
                        
                        # 3. Error Map - compute on CPU to avoid GPU memory spike
                        error_cpu = torch.abs(image_cpu - gt_image_cpu).mean(dim=0, keepdim=True)
                        error_colored_cpu = torch.cat([error_cpu, 1.0 - error_cpu, 1.0 - error_cpu], dim=0)
                        tb_writer.add_images(config['name'] + "_view_{}/error_map".format(viewpoint.image_name), 
                                           error_colored_cpu[None], global_step=iteration)
                        
                        # 4. Gaussian density heat map - compute on CPU
                        with torch.no_grad():
                            img_std_cpu = torch.std(image_cpu, dim=0, keepdim=True)
                            density_proxy_cpu = torch.clamp(img_std_cpu * 5.0, 0.0, 1.0)
                            density_colored_cpu = torch.cat([
                                density_proxy_cpu,
                                torch.zeros_like(density_proxy_cpu),
                                1.0 - density_proxy_cpu
                            ], dim=0)
                            
                        tb_writer.add_images(config['name'] + "_view_{}/gaussian_density_heatmap".format(viewpoint.image_name), 
                                           density_colored_cpu[None], global_step=iteration)
                        
                        # Cleanup ALL visualization tensors
                        del image_cpu, gt_image_cpu, error_cpu, error_colored_cpu, img_std_cpu, density_proxy_cpu, density_colored_cpu
                    
                    # Phase 2d: Wrap metric computation in no_grad() to prevent autograd graph from keeping tensors
                    with torch.no_grad():
                        l1_test += l1_loss(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()
                        ssim_test += ssim(image, gt_image).mean().double()
                        if opt.enable_lpips:
                            lpips_test += lpips_fn(image, gt_image).mean().double()
                    
                    # Phase 2: Immediately release GT image from GPU after metrics computed
                    del gt_image
                    
                    # Comprehensive cleanup after each viewpoint
                    del render_pkg, image
                    
                    # Force cache clear every 3 views instead of 5
                    if idx % 3 == 0:
                        torch.cuda.empty_cache()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                if opt.enable_lpips:
                    lpips_test /= len(config['cameras'])
                
                # Create Rich table for metrics
                table = Table(title=f"[bold cyan]Evaluation Results - Iteration {iteration}[/bold cyan]", 
                             show_header=True, header_style="bold magenta", border_style="cyan")
                table.add_column("Dataset", style="cyan", justify="center")
                table.add_column("L1 Loss", style="yellow", justify="right")
                table.add_column("PSNR (dB)", style="green", justify="right")
                table.add_column("SSIM", style="magenta", justify="right")
                if opt.enable_lpips:
                    table.add_column("LPIPS", style="red", justify="right")
                
                if opt.enable_lpips:
                    table.add_row(
                        config['name'].upper(),
                        f"{l1_test:.6f}",
                        f"{psnr_test:.2f}",
                        f"{ssim_test:.4f}",
                        f"{lpips_test:.4f}"
                    )
                else:
                    table.add_row(
                        config['name'].upper(),
                        f"{l1_test:.6f}",
                        f"{psnr_test:.2f}",
                        f"{ssim_test:.4f}"
                    )
                
                log.print_table(table)
                
                # Store test metrics globally for PLY analysis (only for 'test' set)
                if config['name'] == 'test':
                    last_test_metrics['psnr'] = float(psnr_test)
                    last_test_metrics['ssim'] = float(ssim_test)
                    last_test_metrics['lpips'] = float(lpips_test) if opt.enable_lpips else None
                
                if tb_writer:
                    prefix = '2_test' if config['name'] == 'test' else '1_train'
                    tb_writer.add_scalar(f"{prefix}/loss_viewpoint_l1", l1_test, iteration)
                    tb_writer.add_scalar(f"{prefix}/loss_viewpoint_psnr", psnr_test, iteration)
                    tb_writer.add_scalar(f"{prefix}/loss_viewpoint_ssim", ssim_test, iteration)
                    if opt.enable_lpips:
                        tb_writer.add_scalar(f"{prefix}/loss_viewpoint_lpips", lpips_test, iteration)
        
        # Cleanup LPIPS model (outside loop - both test and train configs share same model)
        if opt.enable_lpips and lpips_fn is not None:
            del lpips_fn
        
        # Force Python garbage collection after evaluation
        gc.collect()
        log.log(f"[green]✓ Evaluation complete at iteration {iteration}[/green]")

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('1_train/total_points', scene.gaussians.get_xyz.shape[0], iteration)
            # FLUSH after evaluation to free image/histogram buffers
            tb_writer.flush()
        torch.cuda.empty_cache()

def enhanced_training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene, renderFunc, renderArgs, gaussians, opt, dataset, nvml_handle=None, lpips_fn=None, log_metrics=True):
    """Enhanced training report with additional performance and Gaussian statistics."""
    
    # Call base training report (pass gaussians for checkpoint/reload logic, conditionally log metrics)
    training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene, renderFunc, renderArgs, opt, dataset, gaussians, lpips_fn, log_metrics=log_metrics)
    
    if not tb_writer:
        return
    
    # === Performance Metrics ===
    if elapsed > 0:
        fps = 1000.0 / elapsed  # elapsed is in ms (keep for console output, don't log to TensorBoard)
    
    # GPU Memory (VRAM)
    if torch.cuda.is_available():
        vram_allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        vram_reserved_gb = torch.cuda.memory_reserved() / (1024**3)
        tb_writer.add_scalar('4_performance/vram_allocated_GB', vram_allocated_gb, iteration)
        tb_writer.add_scalar('4_performance/vram_reserved_GB', vram_reserved_gb, iteration)
        
        # Comprehensive GPU monitoring with pynvml (every 50 iterations for speed)
        if iteration % 50 == 0 and nvml_handle is not None:
            try:
                # Utilization (compute and memory)
                util = pynvml.nvmlDeviceGetUtilizationRates(nvml_handle)
                tb_writer.add_scalar('4_performance/gpu_compute_percent', util.gpu, iteration)
                tb_writer.add_scalar('4_performance/gpu_memory_percent', util.memory, iteration)
                
                # Power
                power_draw = pynvml.nvmlDeviceGetPowerUsage(nvml_handle) / 1000.0  # mW to W
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(nvml_handle) / 1000.0  # mW to W
                tb_writer.add_scalar('4_performance/gpu_power_watts', power_draw, iteration)
                tb_writer.add_scalar('4_performance/gpu_power_percent', 100.0 * power_draw / power_limit if power_limit > 0 else 0, iteration)
                
                # Temperature
                temp_gpu = pynvml.nvmlDeviceGetTemperature(nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
                tb_writer.add_scalar('4_performance/gpu_temperature_celsius', temp_gpu, iteration)
                
                # Fan speed (if available)
                try:
                    fan_speed = pynvml.nvmlDeviceGetFanSpeed(nvml_handle)
                    if fan_speed > 0:
                        tb_writer.add_scalar('4_performance/fan_speed_percent', fan_speed, iteration)
                except pynvml.NVMLError_NotSupported:
                    pass  # Fan control not supported on this GPU
                
                # Clock speeds
                clock_graphics = pynvml.nvmlDeviceGetClockInfo(nvml_handle, pynvml.NVML_CLOCK_GRAPHICS)
                clock_sm = pynvml.nvmlDeviceGetClockInfo(nvml_handle, pynvml.NVML_CLOCK_SM)
                clock_memory = pynvml.nvmlDeviceGetClockInfo(nvml_handle, pynvml.NVML_CLOCK_MEM)
                tb_writer.add_scalar('4_performance/clock_graphics_mhz', clock_graphics, iteration)
                tb_writer.add_scalar('4_performance/clock_sm_mhz', clock_sm, iteration)
                tb_writer.add_scalar('4_performance/clock_memory_mhz', clock_memory, iteration)
                
                # Performance state (P0-P12)
                pstate = pynvml.nvmlDeviceGetPerformanceState(nvml_handle)
                tb_writer.add_scalar('4_performance/pstate', pstate, iteration)
                
                # Throttle reasons (bit flags)
                throttle_reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(nvml_handle)
                is_hw_throttled = 1.0 if (throttle_reasons & pynvml.nvmlClocksThrottleReasonHwSlowdown) else 0.0
                is_power_throttled = 1.0 if (throttle_reasons & pynvml.nvmlClocksThrottleReasonSwPowerCap) else 0.0
                tb_writer.add_scalar('4_performance/throttle_hw_slowdown', is_hw_throttled, iteration)
                tb_writer.add_scalar('4_performance/throttle_power_cap', is_power_throttled, iteration)
                
            except Exception as e:
                pass  # Skip if pynvml query fails
        
        # PCIe info logged once at startup with main nvidia-smi query (already captured above)
    
    # System RAM usage
    ram_info = psutil.virtual_memory()
    ram_used_gb = ram_info.used / (1024**3)
    ram_percent = ram_info.percent
    tb_writer.add_scalar('4_performance/ram_used_GB', ram_used_gb, iteration)
    tb_writer.add_scalar('4_performance/ram_percent', ram_percent, iteration)
    
    # === Gaussian Statistics ===
    num_gaussians = gaussians.get_xyz.shape[0]
    tb_writer.add_scalar('3_gaussians/total_count_K', num_gaussians / 1000.0, iteration)
    
    # Opacity statistics (detached to prevent gradient retention)
    opacity = gaussians.get_opacity.detach()
    tb_writer.add_scalar('3_gaussians/opacity_mean', opacity.mean().item(), iteration)
    tb_writer.add_scalar('3_gaussians/opacity_max', opacity.max().item(), iteration)
    tb_writer.add_scalar('3_gaussians/opacity_min', opacity.min().item(), iteration)
    
    # Count by opacity range
    low_opacity = (opacity < 0.1).sum().item()
    high_opacity = (opacity > 0.9).sum().item()
    tb_writer.add_scalar('3_gaussians/low_opacity_count_K', low_opacity / 1000.0, iteration)
    tb_writer.add_scalar('3_gaussians/high_opacity_count_K', high_opacity / 1000.0, iteration)
    tb_writer.add_scalar('3_gaussians/low_opacity_percent', 100.0 * low_opacity / num_gaussians, iteration)
    
    # Scale statistics (detached)
    scales = gaussians.get_scaling.detach()
    tb_writer.add_scalar('3_gaussians/scale_mean', scales.mean().item(), iteration)
    tb_writer.add_scalar('3_gaussians/scale_max', scales.max().item(), iteration)
    tb_writer.add_scalar('3_gaussians/scale_min', scales.min().item(), iteration)
    
    # === Histograms (every 1000 iterations to reduce overhead) ===
    if iteration % 1000 == 0:
        # Downsample to max 10K samples to prevent memory accumulation
        num_gaussians = scales.shape[0]
        max_samples = min(10000, num_gaussians)
        sample_indices = torch.randperm(num_gaussians, device='cuda')[:max_samples]
        
        # Scale histogram (downsampled and moved to CPU)
        scale_sample = scales[sample_indices].cpu()
        tb_writer.add_histogram("scene/scale_histogram", scale_sample, iteration)
        del scale_sample
        
        # Rotation magnitude histogram (downsampled)
        rotations = gaussians.get_rotation.detach()
        rotation_mag = torch.norm(rotations[sample_indices], dim=1).cpu()
        tb_writer.add_histogram("scene/rotation_magnitude_histogram", rotation_mag, iteration)
        del rotations, rotation_mag
        
        # XYZ magnitude histogram (downsampled)
        xyz = gaussians.get_xyz.detach()
        xyz_mag = torch.norm(xyz[sample_indices], dim=1).cpu()
        tb_writer.add_histogram("scene/xyz_magnitude_histogram", xyz_mag, iteration)
        del xyz, xyz_mag, sample_indices
        
        # CRITICAL: Flush TensorBoard writer to disk to free memory
        tb_writer.flush()
    
    # Cleanup
    del opacity, scales

def print_final_training_summary(model_path):
    """Display comprehensive training summary with key metrics from TensorBoard logs."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
        
        # Find the most recent TensorBoard event file
        import glob
        event_files = glob.glob(os.path.join(model_path, "events.out.tfevents.*"))
        if not event_files:
            log.log("[yellow]⚠️  No TensorBoard logs found - skipping summary[/yellow]")
            return
        
        # Load the most recent event file
        event_file = max(event_files, key=os.path.getmtime)
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
        
        # Helper function to get final scalar value
        def get_final_value(tag):
            try:
                if tag in ea.Tags()['scalars']:
                    events = ea.Scalars(tag)
                    if events:
                        return events[-1].value
            except:
                pass
            return None
        
        # Helper function to get test set metrics
        def get_test_value(tag):
            try:
                if tag in ea.Tags()['scalars']:
                    events = ea.Scalars(tag)
                    # Get last test iteration value (not the very last which might be train)
                    test_events = [e for e in events if '2_test' in tag or 'test' in tag]
                    if test_events:
                        return test_events[-1].value
                    elif events:
                        return events[-1].value
            except:
                pass
            return None
        
        log.log("[cyan bold]Training Summary - Key Metrics[/cyan bold]")
        
        # === Table 1: Model Quality Metrics ===
        # Try to get test metrics, fallback to train if not available
        test_psnr = get_test_value('2_test/loss_viewpoint_psnr')
        test_l1 = get_test_value('2_test/loss_viewpoint_l1')
        train_psnr = get_final_value('1_train/loss_viewpoint_psnr')
        train_l1 = get_final_value('1_train/loss_viewpoint_l1')
        
        # Use train metrics if test not available
        display_psnr = test_psnr if test_psnr else train_psnr
        display_l1 = test_l1 if test_l1 else train_l1
        metric_source = "Test Set" if test_psnr else "Train Set"
        
        quality_table = Table(title=f"[bold green]Model Quality ({metric_source})[/bold green]", 
                             show_header=True, header_style="bold magenta", border_style="green", width=86)
        quality_table.add_column("Metric", style="cyan", justify="left", width=30)
        quality_table.add_column("Final Value", style="yellow bold", justify="right", width=20)
        quality_table.add_column("Target/Reference", style="dim", justify="right", width=36)
        
        # PSNR interpretation
        psnr_quality = ""
        if display_psnr:
            if display_psnr >= 35:
                psnr_quality = "✨ Excellent"
            elif display_psnr >= 30:
                psnr_quality = "✅ Very Good"
            elif display_psnr >= 25:
                psnr_quality = "⚡ Good"
            else:
                psnr_quality = "⚠️  Fair"
        
        quality_table.add_row(
            "PSNR (Peak Signal-to-Noise Ratio)",
            f"{display_psnr:.2f} dB" if display_psnr else "N/A",
            f"{psnr_quality} | >30dB = very good, >35dB = excellent"
        )
        quality_table.add_row(
            "L1 Loss",
            f"{display_l1:.6f}" if display_l1 else "N/A",
            "Lower is better | Pixel error magnitude"
        )
        

        complexity_table = Table(title="[bold blue]Model Complexity[/bold blue]", 
                                show_header=True, header_style="bold magenta", border_style="blue", width=86)
        complexity_table.add_column("Metric", style="cyan", justify="left", width=30)
        complexity_table.add_column("Final Value", style="yellow bold", justify="right", width=20)
        complexity_table.add_column("Notes", style="dim", justify="right", width=36)
        
        total_gaussians = get_final_value('1_train/total_points')
        opacity_mean = get_final_value('3_gaussians/opacity_mean')
        
        if total_gaussians:
            complexity_table.add_row(
                "Total Gaussians",
                f"{int(total_gaussians):,}",
                f"({total_gaussians/1000:.1f}K) | More = finer detail but slower"
            )
        
        if opacity_mean:
            complexity_table.add_row(
                "Average Opacity",
                f"{opacity_mean:.3f}",
                "0=transparent, 1=opaque | Healthy range: 0.3-0.7"
            )
        
        log.log()
        log.print_table(complexity_table)
        
        # Performance table
        perf_table = Table(title="[bold magenta]Training Performance[/bold magenta]", 
                          show_header=True, header_style="bold magenta", border_style="magenta", width=86)
        perf_table.add_column("Metric", style="cyan", justify="left", width=30)
        perf_table.add_column("Value", style="yellow bold", justify="right", width=20)
        perf_table.add_column("Status", style="dim", justify="right", width=36)
        
        final_loss = get_final_value('1_train/total_loss')
        avg_iter_time = get_final_value('1_train/iter_time')
        max_vram = get_final_value('4_performance/vram_allocated_GB')
        
        if final_loss:
            perf_table.add_row("Final Training Loss", f"{final_loss:.6f}", "Lower = better convergence")
        
        if avg_iter_time:
            fps = 1000.0 / avg_iter_time if avg_iter_time > 0 else 0
            perf_table.add_row(
                "Iteration Speed",
                f"{fps:.2f} it/s",
                f"({avg_iter_time:.1f} ms/iter)"
            )
        
        if max_vram:
            vram_status = "✅ Efficient" if max_vram < 14 else "⚠️  High" if max_vram < 15.5 else "🔴 Near Limit"
            perf_table.add_row(
                "Peak VRAM Usage",
                f"{max_vram:.2f} GB",
                f"{vram_status} | Target: <14GB for safety"
            )
        
        log.log()
        log.print_table(perf_table)
        
        # === Output Files ===
        log.print_table(perf_table)
        
        # === Output Files ===tual saved files
        import glob
        ply_files = sorted(glob.glob(os.path.join(model_path, "point_cloud/iteration_*/point_cloud.ply")))
        checkpoint_files = sorted(glob.glob(os.path.join(model_path, "chkpnt*.pth")))
        
        if ply_files:
            for ply in ply_files[-3:]:  # Show last 3
                rel_path = os.path.relpath(ply, model_path)
                abs_path = os.path.abspath(ply)
                log.log(f"  • [link=file://{abs_path}]{rel_path}[/link]")
        
        if checkpoint_files:
            for chk in checkpoint_files[-3:]:  # Show last 3
                rel_path = os.path.relpath(chk, model_path)
                abs_path = os.path.abspath(chk)
                log.log(f"  • [link=file://{abs_path}]{rel_path}[/link]")
        
        # Quick quality assessment (use display_psnr which falls back to train if test not available)
        if display_psnr and display_psnr >= 30:
            log.log("[green]✅ Training completed successfully with good quality results![/green]")
        elif display_psnr and display_psnr >= 25:
            log.log("[yellow]⚡ Training completed. Results are decent - consider more iterations or adjusting parameters.[/yellow]")
        elif display_psnr:
            log.log("[yellow]⚠️  Training completed but quality is lower than expected. Check dataset quality or training parameters.[/yellow]")
        else:
            log.log("Training completed. Check TensorBoard for detailed metrics.")
        
    except ImportError:
        log.log("[yellow]⚠️  TensorBoard not available - install with: pip install tensorboard[/yellow]")
    except Exception as e:
        log.log(f"[yellow]⚠️  Could not generate training summary: {e}[/yellow]")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--experiment_name", type=str, default=None, help="Name for this training run (appended to output path)")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    # Auto-enable --eval when --eval_only is set (eval_only requires test cameras)
    if args.eval_only and not args.eval:
        args.eval = True
        log.log("[yellow]ℹ️  Auto-enabled --eval (required for --eval_only to have test cameras)[/yellow]")
    
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # Extract parameters and pass experiment_name to dataset
    dataset_params = lp.extract(args)
    # Preserve experiment_name from main args
    dataset_params.experiment_name = args.experiment_name
    
    training(dataset_params, op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    log.log("[green bold]Training complete.[/green bold]")
    
    # Display final training summary (use dataset_params.model_path which was set in training)
    print_final_training_summary(dataset_params.model_path)
