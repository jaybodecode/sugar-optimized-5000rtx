import os
import sys
import time
import psutil
import numpy as np
import torch
import torch.utils.checkpoint
import open3d as o3d
import hashlib
import lpips
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.transforms import quaternion_apply, quaternion_invert
from sugar_scene.gs_model import GaussianSplattingWrapper, fetchPly
from sugar_scene.cameras import GSCamera
from sugar_scene.sugar_model import SuGaR
from sugar_scene.sugar_optimizer import OptimizationParams, SuGaROptimizer
from sugar_scene.sugar_densifier import SuGaRDensifier
from sugar_utils.loss_utils import ssim, l1_loss, l2_loss

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from torch.utils.tensorboard import SummaryWriter
import time


def depths_to_points(view, depthmap):
    """Comes from 2DGS.

    Args:
        view (_type_): _description_
        depthmap (_type_): _description_

    Returns:
        _type_: _description_
    """
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W) / 2],
        [0, H / 2, 0, (H) / 2],
        [0, 0, 0, 1]]).float().cuda().T
    projection_matrix = c2w.T @ view.full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3,:3].T
    
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points


def depth2normal_2dgs(view, depth):
    """Comes from 2DGS.
    
        view: view camera
        depth: depthmap 
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output


def depth_normal_consistency_loss(
    depth:torch.Tensor, 
    normal:torch.Tensor,
    camera,
    scale_rendered_normals=False,
    return_normal_maps=False
):
    """_summary_

    Args:
        depth (torch.Tensor): Has shape (1, height, width).
        normal (torch.tensor): Has shape (3, height, width). Should be in view space.
        opacity (torch.Tensor): Has shape (1, height, width).
        camera (GSCamera): _description_

    Returns:
        _type_: _description_
    """
    
    # Compute the normals from the depth map in world space.
    normal_from_depth = depth2normal_2dgs(camera, depth)
    
    # Transform the normals from the depth map to the view space (COLMAP convention).
    normal_from_depth = (normal_from_depth @ camera.world_view_transform[:3,:3]).permute(2, 0, 1)
    
    if scale_rendered_normals:
        # We normalize the normals to have the same scale as the normals from the depth map.
        normal_view = ((normal - normal.mean()) / normal.std()) * normal_from_depth.std() + normal_from_depth.mean()
    else:
        normal_view = normal

    # Compute the error between the normals from the depth map and the rendered normals.    
    normal_error = (1 - (normal_view * normal_from_depth).sum(dim=0))
    
    if return_normal_maps:
        return normal_error, normal_view, normal_from_depth    
    return normal_error.mean()


def coarse_training_with_density_regularization_and_dn_consistency(args):
    CONSOLE = Console(width=120)

    # ====================Parameters====================

    num_device = args.gpu
    detect_anomaly = False

    # -----Data parameters-----
    downscale_resolution_factor = 1  # 2, 4

    # -----Model parameters-----
    use_eval_split = True
    n_skip_images_for_eval_split = 8

    freeze_gaussians = False
    initialize_from_trained_3dgs = True  # True or False
    if initialize_from_trained_3dgs:
        prune_at_start = False
        start_pruning_threshold = 0.5
    no_rendering = freeze_gaussians

    n_points_at_start = None  # If None, takes all points in the SfM point cloud

    learnable_positions = True  # True in 3DGS
    use_same_scale_in_all_directions = False  # Should be False
    sh_levels = 4  

        
    # -----Radiance Mesh-----
    triangle_scale=1.
    
        
    # -----Rendering parameters-----
    compute_color_in_rasterizer = True  # TODO: Try True

        
    # -----Optimization parameters-----

    # Learning rates and scheduling
    num_iterations = args.coarse_iterations if hasattr(args, 'coarse_iterations') else 15_000

    spatial_lr_scale = None
    position_lr_init=0.00016
    position_lr_final=0.0000016
    position_lr_delay_mult=0.01
    position_lr_max_steps=30_000
    feature_lr=0.0025
    opacity_lr=0.05
    scaling_lr=0.005
    rotation_lr=0.001
        
    # Densifier and pruning
    heavy_densification = False
    if initialize_from_trained_3dgs:
        densify_from_iter = 500 + 99999 # 500  # Maybe reduce this, since we have a better initialization?
        densify_until_iter = 7000 - 7000 # 7000
    else:
        densify_from_iter = 500 # 500  # Maybe reduce this, since we have a better initialization?
        densify_until_iter = 7000 # 7000

    if heavy_densification:
        densification_interval = 50  # 100
        opacity_reset_interval = 3000  # 3000
        
        densify_grad_threshold = 0.0001  # 0.0002
        densify_screen_size_threshold = 20
        prune_opacity_threshold = 0.005
        densification_percent_distinction = 0.01
    else:
        densification_interval = 100  # 100
        opacity_reset_interval = 3000  # 3000
        
        densify_grad_threshold = 0.0002  # 0.0002
        densify_screen_size_threshold = 20
        prune_opacity_threshold = 0.005
        densification_percent_distinction = 0.01

    # Data processing and batching
    n_images_to_use_for_training = -1  # If -1, uses all images

    train_num_images_per_batch = 1  # 1 for full images
    
    # Gradient checkpointing (Phase 2 optimization)
    use_gradient_checkpointing = args.use_gradient_checkpointing if hasattr(args, 'use_gradient_checkpointing') else True

    # Loss functions
    loss_function = 'l1+dssim'  # 'l1' or 'l2' or 'l1+dssim'
    if loss_function == 'l1+dssim':
        dssim_factor = 0.2
        
    # Depth-Normal consistency (can be disabled for -r density mode)
    use_dn_consistency = args.use_dn_consistency if hasattr(args, 'use_dn_consistency') else True
    enforce_depth_normal_consistency = use_dn_consistency
    if enforce_depth_normal_consistency:
        start_dn_consistency_from = 9000  # 7000
        dn_consistency_factor = 0.05  # 0.1

    # Regularization
    enforce_entropy_regularization = True
    if enforce_entropy_regularization:
        start_entropy_regularization_from = 7000
        end_entropy_regularization_at = 9000  # TODO: Change
        entropy_regularization_factor = 0.1
            
    regularize_sdf = True
    if regularize_sdf:
        beta_mode = 'average'  # 'learnable', 'average' or 'weighted_average'
        
        start_sdf_regularization_from = 9000
        regularize_sdf_only_for_gaussians_with_high_opacity = False
        if regularize_sdf_only_for_gaussians_with_high_opacity:
            sdf_regularization_opacity_threshold = 0.5
            
        use_sdf_estimation_loss = True
        enforce_samples_to_be_on_surface = False
        if use_sdf_estimation_loss or enforce_samples_to_be_on_surface:
            sdf_estimation_mode = 'density'  # 'sdf' or 'density'
            # sdf_estimation_factor = 0.2  # 0.1 or 0.2?
            samples_on_surface_factor = 0.2  # 0.05
            
            squared_sdf_estimation_loss = False
            squared_samples_on_surface_loss = False
            
            normalize_by_sdf_std = False  # False
            
            start_sdf_estimation_from = 9000  # 7000
            
            sample_only_in_gaussians_close_to_surface = True
            close_gaussian_threshold = 2.  # 2.
            
            use_projection_as_estimation = True
            if use_projection_as_estimation:
                sample_only_in_gaussians_close_to_surface = False
            
            backpropagate_gradients_through_depth = True  # True
            
        use_sdf_better_normal_loss = True
        if use_sdf_better_normal_loss:
            start_sdf_better_normal_from = 9000
            # sdf_better_normal_factor = 0.2  # 0.1 or 0.2?
            sdf_better_normal_gradient_through_normal_only = True
        
        density_factor = 1. / 16. # Should be equal to 1. / regularity_knn
        if (use_sdf_estimation_loss or enforce_samples_to_be_on_surface) and sdf_estimation_mode == 'density':
            density_factor = 1.
        density_threshold = 1.  # 0.5 * density_factor
        n_samples_for_sdf_regularization = 250_000  # Reduced from 1M for VRAM optimization (4-6 GB savings, zero quality impact)
        sdf_sampling_scale_factor = 1.5
        sdf_sampling_proportional_to_volume = False
        
    bind_to_surface_mesh = False
    if bind_to_surface_mesh:
        learn_surface_mesh_positions = True
        learn_surface_mesh_opacity = True
        learn_surface_mesh_scales = True
        n_gaussians_per_surface_triangle=6  # 1, 3, 4 or 6
        
        use_surface_mesh_laplacian_smoothing_loss = True
        if use_surface_mesh_laplacian_smoothing_loss:
            surface_mesh_laplacian_smoothing_method = "uniform"  # "cotcurv", "cot", "uniform"
            surface_mesh_laplacian_smoothing_factor = 5.  # 0.1
        
        use_surface_mesh_normal_consistency_loss = True
        if use_surface_mesh_normal_consistency_loss:
            surface_mesh_normal_consistency_factor = 0.1  # 0.1
            
        densify_from_iter = 999_999
        densify_until_iter = 0
        position_lr_init=0.00016 * 0.01
        position_lr_final=0.0000016 * 0.01
        scaling_lr=0.005
    else:
        surface_mesh_to_bind_path = None
        
    if regularize_sdf:
        regularize = True
        regularity_knn = 16  # 8 until now
        # regularity_knn = 8
        regularity_samples = 10000  # Changed from -1 (all points) for 100x speedup with minimal quality loss
        reset_neighbors_every = 500  # 500 until now
        regularize_from = 7000  # 0 until now
        start_reset_neighbors_from = 7000+1  # 0 until now (should be equal to regularize_from + 1?)
        prune_when_starting_regularization = False
    else:
        regularize = False
        regularity_knn = 0
    if bind_to_surface_mesh:
        regularize = False
        regularity_knn = 0
        
    # Opacity management
    prune_low_opacity_gaussians_at = [9000]
    if bind_to_surface_mesh:
        prune_low_opacity_gaussians_at = [999_999]
    prune_hard_opacity_threshold = 0.5

    # Warmup
    do_resolution_warmup = False  # DISABLED: Progressive resolution warmup requires rescale_output_resolution method not yet implemented in CamerasWrapper
    if do_resolution_warmup:
        resolution_warmup_every = 500
        current_resolution_factor = downscale_resolution_factor * 4.
    else:
        current_resolution_factor = downscale_resolution_factor

    do_sh_warmup = True  # Should be True
    if initialize_from_trained_3dgs:
        do_sh_warmup = False
        sh_levels = 4  # nerfmodel.gaussians.active_sh_degree + 1
        CONSOLE.print("Changing sh_levels to match the loaded model:", sh_levels)
    if do_sh_warmup:
        sh_warmup_every = 1000
        current_sh_levels = 1
    else:
        current_sh_levels = sh_levels
        

    # -----Log and save-----
    print_loss_every_n_iterations = 10  # Log frequently for detailed TensorBoard graphs
    save_model_every_n_iterations = args.checkpoint_interval if hasattr(args, 'checkpoint_interval') else 1_000
    # save_milestones = [9000, 12_000, 15_000]
    save_milestones = args.checkpoint_milestones if hasattr(args, 'checkpoint_milestones') else [7000, 9000, 12_000, 15_000]
    test_iterations = args.test_iterations if hasattr(args, 'test_iterations') else [7000, 10000, 15000]

    # ====================End of parameters====================

    if args.output_dir is None:
        if len(args.scene_path.split("/")[-1]) > 0:
            args.output_dir = os.path.join("./output/coarse", args.scene_path.split("/")[-1])
        else:
            args.output_dir = os.path.join("./output/coarse", args.scene_path.split("/")[-2])
            
    source_path = args.scene_path
    gs_checkpoint_path = args.checkpoint_path
    iteration_to_load = args.iteration_to_load    
    
    sdf_estimation_factor = args.estimation_factor
    sdf_better_normal_factor = args.normal_factor
    
    sugar_checkpoint_path = f'sugarcoarse_3Dgs{iteration_to_load}_densityestimXX_sdfnormYY/'
    sugar_checkpoint_path = os.path.join(args.output_dir, sugar_checkpoint_path)
    sugar_checkpoint_path = sugar_checkpoint_path.replace(
        'XX', str(sdf_estimation_factor).replace('.', '')
        ).replace(
            'YY', str(sdf_better_normal_factor).replace('.', '')
            )
    
    use_eval_split = args.eval
    use_white_background = args.white_background
    
    ply_path = os.path.join(source_path, "sparse/0/points3D.ply")
    
    # Create configuration summary table
    CONSOLE.print("\n" + "═" * 80)
    config_table = Table(title="⚙️  Configuration Summary", show_header=True, header_style="bold cyan")
    config_table.add_column("Setting", style="cyan", width=30)
    config_table.add_column("Value", style="green", width=45)
    
    config_table.add_row("Source Path", source_path)
    config_table.add_row("  └─ Files/Folders", str(len(os.listdir(source_path))))
    config_table.add_row("3DGS Checkpoint", gs_checkpoint_path)
    config_table.add_row("  └─ Files/Folders", str(len(os.listdir(gs_checkpoint_path))))
    config_table.add_row("  └─ Iteration", f"{iteration_to_load:,}")
    config_table.add_row("SuGaR Output Path", sugar_checkpoint_path)
    config_table.add_row("Output Directory", args.output_dir)
    config_table.add_row("", "")  # Spacer
    if enforce_depth_normal_consistency:
        config_table.add_row("Depth-Normal Factor", f"{dn_consistency_factor:.3f}")
    else:
        config_table.add_row("Depth-Normal Factor", "✗ Disabled (-r density)")
    config_table.add_row("SDF Estimation Factor", f"{sdf_estimation_factor:.3f}")
    config_table.add_row("SDF Better Normal Factor", f"{sdf_better_normal_factor:.3f}")
    config_table.add_row("", "")  # Spacer
    config_table.add_row("Eval Split", "✓ Yes" if use_eval_split else "✗ No")
    config_table.add_row("White Background", "✓ Yes" if use_white_background else "✗ No")
    
    CONSOLE.print(config_table)
    CONSOLE.print("═" * 80)
    
    # Regularization mode note
    if not enforce_depth_normal_consistency:
        CONSOLE.print("\n[bold cyan]ℹ️  Density Regularization Mode[/bold cyan]")
        CONSOLE.print("[dim]  → Depth-normal consistency disabled (uses density-based regularization only)[/dim]")
        CONSOLE.print("[dim]  → Recommended for 16GB VRAM[/dim]")
        CONSOLE.print("[dim]  → Excellent quality for production meshes[/dim]")
        CONSOLE.print("[dim]  → For best quality with 24GB+ VRAM, use -r dn_consistency[/dim]")
        CONSOLE.print()
    
    # VRAM Optimization Warning
    use_full_res_normals = args.full_res_normals if hasattr(args, 'full_res_normals') else False
    
    CONSOLE.print()
    if not use_full_res_normals:
        vram_panel = Panel(
            "[yellow]Depth-normal maps rendering at [bold cyan]half resolution[/bold cyan][/yellow]\n"
            "\n"
            "💾 [green]Saves:[/green] 4-5GB VRAM\n"
            "📊 [yellow]Impact:[/yellow] May reduce PSNR slightly (~0.1-0.2 dB)\n"
            "🎯 [dim]For best quality:[/dim] [green bold]--full_res_normals True[/green bold] [dim](requires 24GB+ VRAM)[/dim]",
            title="⚠️  VRAM Optimization Active",
            border_style="yellow",
            expand=False
        )
        CONSOLE.print(vram_panel)
    else:
        vram_panel = Panel(
            "[green]Rendering at [bold]full resolution[/bold][/green]\n"
            "\n"
            "📊 [green]Quality:[/green] Maximum (best PSNR)\n"
            "💾 [yellow]VRAM Usage:[/yellow] +4-5GB compared to half-res\n"
            "⚡ [dim]Speed:[/dim] Slower rendering during training",
            title="✅ Full Resolution Mode",
            border_style="green",
            expand=False
        )
        CONSOLE.print(vram_panel)
    
    # Initialize TensorBoard writer
    tb_log_dir = os.path.join(sugar_checkpoint_path, 'tensorboard')
    tb_writer = SummaryWriter(log_dir=tb_log_dir)
    
    # Setup tensor profiling log file if enabled
    tensor_profile_log = None
    if hasattr(args, 'profile_tensors') and args.profile_tensors:
        log_dir = os.path.join(sugar_checkpoint_path, 'profiles')
        os.makedirs(log_dir, exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        tensor_profile_log = os.path.join(log_dir, f'tensor_profile_{timestamp}.log')
        CONSOLE.print(f"[yellow]📊 Tensor profiling enabled - logs: {tensor_profile_log}[/yellow]")
    
    # Determine TensorBoard logdir for comparison display
    # If using new _mesh structure, show parent path to compare all experiments
    parent_dir = os.path.dirname(sugar_checkpoint_path)
    if parent_dir and os.path.basename(parent_dir).endswith('_mesh'):
        # Using new naming scheme - show parent directory for run comparison
        tb_comparison_dir = parent_dir
        comparison_note = " (compare all experiments for this mesh)"
    else:
        # Using old naming or custom path - show specific directory
        tb_comparison_dir = tb_log_dir
        comparison_note = ""
    
    # Show TensorBoard command (easy to copy-paste - outside panel for better copy-paste)
    CONSOLE.print()
    CONSOLE.print("[yellow bold]📋 TensorBoard - Copy-paste this command in another terminal:[/yellow bold]")
    CONSOLE.print(f"tensorboard --logdir \"{tb_comparison_dir}\" --port 6007 --bind_all")
    CONSOLE.print()
    CONSOLE.print(f"[dim]Then open:[/dim] [cyan]http://localhost:6007[/cyan]")
    if comparison_note:
        CONSOLE.print(f"[dim]Note:{comparison_note}[/dim]")
    CONSOLE.print()
    CONSOLE.print(f"[cyan]Logs:[/cyan] [dim]{tb_log_dir}[/dim]")
    CONSOLE.print()
    
    # Show available metrics in panel
    metrics_panel = Panel(
        "[bold]Available Metrics:[/bold]\n"
        "  • [cyan]Loss/train[/cyan] - Training loss (target: 0.17 → 0.05)\n"
        "  • [cyan]Loss/test[/cyan] - Validation loss\n"
        "  • [cyan]VRAM/allocated[/cyan] - GPU memory usage\n"
        "  • [cyan]Parameters/*[/cyan] - Model parameter statistics\n"
        "  • [cyan]Speed/iteration_time[/cyan] - Training speed",
        title="📊 TensorBoard Metrics",
        border_style="cyan",
        expand=False
    )
    CONSOLE.print(metrics_panel)
    CONSOLE.print()
    
    # Add training guide as text in TensorBoard
    training_guide = """
    ## Training Expectations (Garden Scene, 4.9M Gaussians)
    
    **Loss (should decrease ↓):**
    - Start: ~0.17 (iteration 7000)
    - Target: ~0.05 (iteration 15000)
    - Test Loss: ~0.28-0.29 (expected to be higher than training)
    
    **Parameters (should stabilize):**
    - Scales mean: ~0.015 ± 0.02
    - Opacities mean: ~0.32-0.35 ± 0.32
    - Points mean: ~0.77 ± 8.24
    
    **VRAM (should stay stable):**
    - Expected: 13-14 GB (with all optimizations)
    - Alert if: >15 GB (may indicate memory leak)
    
    **Known Issues:**
    - ⚠️ Test/PSNR_BUGGY_IGNORE: Shows ~8-9 dB (incorrect tensor format)
    - ✓ Test/loss is correct metric to monitor
    
    **Baseline Comparison:**
    - Mip-splatting PSNR: 28.48 dB (training), 0.0247 loss
    - SuGaR adds regularization, so higher loss is expected
    """
    tb_writer.add_text('Training_Guide', training_guide, 0)
    
    # Setup device
    torch.cuda.set_device(num_device)
    CONSOLE.print("Using device:", num_device)
    device = torch.device(f'cuda:{num_device}')
    
    # Initialize LPIPS model for test evaluation (if test_iterations specified)
    test_iterations = args.test_iterations if hasattr(args, 'test_iterations') else []
    if test_iterations:
        CONSOLE.print("[cyan]⏳ Initializing LPIPS model for test evaluation...[/cyan]")
        lpips_fn = lpips.LPIPS(net='vgg').to(device)
        CONSOLE.print("[green]✓ LPIPS model ready[/green]")
    else:
        lpips_fn = None
    
    # Only show memory summary if there's actual memory allocated
    if torch.cuda.memory_allocated() > 0:
        CONSOLE.print(torch.cuda.memory_summary())
    
    torch.autograd.set_detect_anomaly(detect_anomaly)
    
    # Creates save directory if it does not exist
    os.makedirs(sugar_checkpoint_path, exist_ok=True)
    
    # ====================Load NeRF model and training data====================

    # Load Gaussian Splatting checkpoint 
    CONSOLE.print(f"\nLoading config {gs_checkpoint_path}...")
    if use_eval_split:
        CONSOLE.print("Performing train/eval split...")
    
    t_load_model_start = time.time()
    nerfmodel = GaussianSplattingWrapper(
        source_path=source_path,
        output_path=gs_checkpoint_path,
        iteration_to_load=iteration_to_load,
        load_gt_images=True,
        eval_split=use_eval_split,
        eval_split_interval=n_skip_images_for_eval_split,
        white_background=use_white_background,
        )
    CONSOLE.print(f"⏱️  Model and images loaded in {time.time() - t_load_model_start:.2f}s")

    CONSOLE.print(f'{len(nerfmodel.training_cameras)} training images detected.')
    CONSOLE.print(f'The model has been trained for {iteration_to_load} steps.')

    if downscale_resolution_factor != 1:
       nerfmodel.downscale_output_resolution(downscale_resolution_factor)
    CONSOLE.print(f'\nCamera resolution scaled to '
          f'{nerfmodel.training_cameras.gs_cameras[0].image_height} x '
          f'{nerfmodel.training_cameras.gs_cameras[0].image_width}'
          )

    # ====================Check for resume checkpoint first====================
    resume_checkpoint_path = args.resume_checkpoint if hasattr(args, 'resume_checkpoint') else None
    
    if resume_checkpoint_path:
        if not os.path.exists(resume_checkpoint_path):
            CONSOLE.print(f"\n[bold red]❌ ERROR: Resume checkpoint not found![/bold red]")
            CONSOLE.print(f"[red]Path:[/red] {resume_checkpoint_path}")
            
            # Try to find available checkpoints and suggest them
            # Look in common locations
            search_paths = [
                './output/coarse',
                os.path.join(os.path.dirname(resume_checkpoint_path)),  # Try the specified directory
            ]
            
            available_checkpoints = []
            for search_dir in search_paths:
                if os.path.exists(search_dir):
                    for root, dirs, files in os.walk(search_dir):
                        for file in files:
                            if file.endswith('.pt') and file[0].isdigit():  # Iteration checkpoints like 7000.pt
                                full_path = os.path.join(root, file)
                                available_checkpoints.append(full_path)
            
            if available_checkpoints:
                # Remove duplicates and sort by iteration number (not alphabetically)
                def get_iteration_number(path):
                    """Extract iteration number from checkpoint path like .../15000.pt"""
                    try:
                        filename = os.path.basename(path)
                        return int(filename.replace('.pt', ''))
                    except:
                        return 0
                
                available_checkpoints = sorted(list(set(available_checkpoints)), key=get_iteration_number)
                
                CONSOLE.print(f"\n[green]💡 Found {len(available_checkpoints)} checkpoint(s):[/green]")
                
                # Show up to 10 most recent
                for cp in available_checkpoints[-10:]:
                    try:
                        rel_path = os.path.relpath(cp, os.getcwd())
                        # Get file size
                        size_mb = os.path.getsize(cp) / (1024 * 1024)
                        iteration = get_iteration_number(cp)
                        CONSOLE.print(f"  • {rel_path} ({size_mb:.0f} MB) [iteration {iteration}]")
                    except:
                        CONSOLE.print(f"  • {cp}")
                
                CONSOLE.print(f"\n[yellow]💡 Try resuming with:[/yellow]")
                # Suggest the highest iteration checkpoint
                example_checkpoint = os.path.relpath(available_checkpoints[-1], os.getcwd())
                CONSOLE.print(f"[cyan]  --resume_checkpoint {example_checkpoint}[/cyan]")
            else:
                CONSOLE.print(f"\n[yellow]⚠ No checkpoints found in ./output/coarse/[/yellow]")
                CONSOLE.print(f"[dim]Training checkpoints are saved every 1000 iterations by default.[/dim]")
            
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint_path}")
    
    is_resuming = resume_checkpoint_path and os.path.exists(resume_checkpoint_path)
    
    if is_resuming:
        CONSOLE.print(f"\n🔄 [bold yellow]Resuming from checkpoint:[/bold yellow] {resume_checkpoint_path}")
        CONSOLE.print("[dim]Skipping initialization - will load full state from checkpoint...[/dim]")
        
        # Load checkpoint to get basic info
        # weights_only=False is safe here since we created these checkpoints ourselves
        checkpoint = torch.load(resume_checkpoint_path, map_location='cpu', weights_only=False)
        
        # We still need points/colors to construct the model, but can skip expensive operations
        # The checkpoint should have saved these, but if not we need minimal initialization
        use_cached_init = False  # Will load full state from checkpoint instead
    else:
        checkpoint = None
    
    # ====================Initialize or load cached SuGaR model====================
    # Create cache key based on source data
    cache_key = hashlib.md5(f"{gs_checkpoint_path}_{iteration_to_load}_{prune_at_start}_{start_pruning_threshold}".encode()).hexdigest()
    cache_file = os.path.join(gs_checkpoint_path, f"sugar_init_cache_{cache_key}.pt")
    
    if is_resuming:
        # When resuming, skip cache check - we'll load from checkpoint
        CONSOLE.print("⏭️  Skipping initialization cache (resuming from checkpoint)")
        use_cached_init = False
    elif os.path.exists(cache_file):
        CONSOLE.print(f"\n🚀 Loading cached SuGaR initialization from {cache_file}...")
        try:
            # weights_only=False is safe here since we created these cache files ourselves
            cached_data = torch.load(cache_file, map_location='cuda', weights_only=False)
            points = cached_data['points']
            colors = cached_data['colors']
            cached_state = cached_data['sugar_state']
            sh_levels = cached_data['sh_levels']
            n_points = len(points)
            use_cached_init = True
            CONSOLE.print(f"✓ Cached initialization loaded. Number of points: {n_points}")
        except Exception as e:
            CONSOLE.print(f"⚠ Failed to load cache: {e}. Re-initializing...")
            use_cached_init = False
    else:
        use_cached_init = False

    # Point cloud
    if is_resuming:
        # When resuming, get minimal info from checkpoint to construct model
        # Full state will be loaded later
        CONSOLE.print("📦 Loading model structure from checkpoint...")
        sh_levels = checkpoint.get('sh_levels', 4)  # Default to 4 if not in checkpoint
        
        # Get point count from checkpoint state dict
        if 'state_dict' in checkpoint and '_points' in checkpoint['state_dict']:
            n_points = checkpoint['state_dict']['_points'].shape[0]
            points = checkpoint['state_dict']['_points'].cuda()
            # Get colors from SH DC component
            if '_sh_coordinates_dc' in checkpoint['state_dict']:
                from sugar_utils.spherical_harmonics import SH2RGB
                colors = SH2RGB(checkpoint['state_dict']['_sh_coordinates_dc'].squeeze(1).cuda())
            else:
                colors = torch.ones_like(points).cuda() * 0.5
        else:
            raise ValueError("Checkpoint missing required state_dict with _points")
        
        CONSOLE.print(f"✓ Loaded {n_points:,} points from checkpoint")
        
    elif not use_cached_init:
        if initialize_from_trained_3dgs:
            with torch.no_grad():    
                print("Initializing model from trained 3DGS...")
                with torch.no_grad():
                    sh_levels = int(np.sqrt(nerfmodel.gaussians.get_features.shape[1]))
                
                from sugar_utils.spherical_harmonics import SH2RGB
                points = nerfmodel.gaussians.get_xyz.detach().float().cuda()
                colors = SH2RGB(nerfmodel.gaussians.get_features[:, 0].detach().float().cuda())
                if prune_at_start:
                    with torch.no_grad():
                        start_prune_mask = nerfmodel.gaussians.get_opacity.view(-1) > start_pruning_threshold
                        points = points[start_prune_mask]
                        colors = colors[start_prune_mask]
                n_points = len(points)
        else:
            CONSOLE.print("\nLoading SfM point cloud...")
            pcd = fetchPly(ply_path)
            points = torch.tensor(pcd.points, device=nerfmodel.device).float().cuda()
            colors = torch.tensor(pcd.colors, device=nerfmodel.device).float().cuda()
        
            if n_points_at_start is not None:
                n_points = n_points_at_start
                pts_idx = torch.randperm(len(points))[:n_points]
                points, colors = points.to(device)[pts_idx], colors.to(device)[pts_idx]
            else:
                n_points = len(points)
                
        CONSOLE.print(f"Point cloud generated. Number of points: {len(points)}")
    
    # Mesh to bind to if needed  TODO
    if bind_to_surface_mesh:
        surface_mesh_to_bind_full_path = os.path.join('./results/meshes/', surface_mesh_to_bind_path)
        CONSOLE.print(f'\nLoading mesh to bind to: {surface_mesh_to_bind_full_path}...')
        o3d_mesh = o3d.io.read_triangle_mesh(surface_mesh_to_bind_full_path)
        CONSOLE.print("Mesh to bind to loaded.")
    else:
        o3d_mesh = None
        learn_surface_mesh_positions = False
        learn_surface_mesh_opacity = False
        learn_surface_mesh_scales = False
        n_gaussians_per_surface_triangle=1
    
    if not regularize_sdf:
        beta_mode = None
        
    # Background tensor if needed
    if use_white_background:
        bg_tensor = torch.ones(3, dtype=torch.float, device=nerfmodel.device)
    else:
        bg_tensor = None
    
    # ====================Initialize SuGaR model====================
    # Construct SuGaR model
    CONSOLE.print("\n🔨 Constructing SuGaR model...")
    t_sugar_start = time.time()
    
    # Skip KNN computation during construction if we have cached KNN data (saves ~4 minutes!)
    should_compute_knn = regularize and not (use_cached_init and 'knn_dists' in cached_state)
    
    # Skip radius initialization if we have cached scales/quaternions (saves ~60s of KNN K=4 computation!)
    should_initialize_radiuses = not use_cached_init
    
    if use_cached_init:
        CONSOLE.print("  [green]✓ Using cached initialization from Gaussian Splatting checkpoint[/green]")
        CONSOLE.print("  [dim]Skipping expensive KNN K=16 computation (~4 min) and radius initialization (~60s)[/dim]")
    else:
        CONSOLE.print("  [yellow]⏳ First run on this scene: Computing KNN K={} (~3-4 min) + radius initialization (~60s)[/yellow]".format(regularity_knn))
        CONSOLE.print("  [dim]Results will be cached in GS checkpoint for future experiments on this scene[/dim]")
    
    try:
        sugar = SuGaR(
            nerfmodel=nerfmodel,
            points=points, #nerfmodel.gaussians.get_xyz.data,
            colors=colors, #0.5 + _C0 * nerfmodel.gaussians.get_features.data[:, 0, :],
            initialize=should_initialize_radiuses,  # Skip if we have cached scales/quaternions
            sh_levels=sh_levels,
            learnable_positions=learnable_positions,
            triangle_scale=triangle_scale,
            keep_track_of_knn=should_compute_knn,  # Only compute if not cached
            knn_to_track=regularity_knn,
            beta_mode=beta_mode,
            freeze_gaussians=freeze_gaussians,
            surface_mesh_to_bind=o3d_mesh,
            surface_mesh_thickness=None,
            learn_surface_mesh_positions=learn_surface_mesh_positions,
            learn_surface_mesh_opacity=learn_surface_mesh_opacity,
            learn_surface_mesh_scales=learn_surface_mesh_scales,
            n_gaussians_per_surface_triangle=n_gaussians_per_surface_triangle,
            )
    except KeyboardInterrupt:
        CONSOLE.print(f"\n\n[bold yellow]⚠️  Model construction interrupted by user (CTRL-C)[/bold yellow]")
        CONSOLE.print("[dim]KNN computation or radius initialization was interrupted.[/dim]")
        CONSOLE.print(f"\n[bold]Exiting gracefully...[/bold]\n")
        raise SystemExit(0)
    
    CONSOLE.print(f"⏱️  SuGaR model constructed in {time.time() - t_sugar_start:.2f}s")
    
    if initialize_from_trained_3dgs:
        if use_cached_init:
            # Load cached parameters
            t_load_start = time.time()
            with torch.no_grad():
                CONSOLE.print("Loading cached Gaussian parameters...")
                sugar._scales[...] = cached_state['scales']
                sugar._quaternions[...] = cached_state['quaternions']
                sugar.all_densities[...] = cached_state['opacities']
                sugar._sh_coordinates_dc[...] = cached_state['sh_dc']
                sugar._sh_coordinates_rest[...] = cached_state['sh_rest']
                
                # Load KNN data if available and restore keep_track_of_knn state
                if 'knn_dists' in cached_state and 'knn_idx' in cached_state and regularize:
                    CONSOLE.print("Loading cached KNN data...")
                    sugar.keep_track_of_knn = True
                    sugar.knn_to_track = regularity_knn
                    sugar.knn_dists = cached_state['knn_dists'].to(sugar.device)
                    sugar.knn_idx = cached_state['knn_idx'].to(sugar.device)
            CONSOLE.print(f"⏱️  Cached parameters loaded in {time.time() - t_load_start:.2f}s")
        elif not is_resuming:  # FIXED: Skip 3DGS initialization when resuming from checkpoint
            with torch.no_grad():            
                CONSOLE.print("Initializing 3D gaussians from 3D gaussians...")
                if prune_at_start:
                    sugar._scales[...] = nerfmodel.gaussians._scaling.detach()[start_prune_mask]
                    sugar._quaternions[...] = nerfmodel.gaussians._rotation.detach()[start_prune_mask]
                    sugar.all_densities[...] = nerfmodel.gaussians._opacity.detach()[start_prune_mask]
                    sugar._sh_coordinates_dc[...] = nerfmodel.gaussians._features_dc.detach()[start_prune_mask]
                    sugar._sh_coordinates_rest[...] = nerfmodel.gaussians._features_rest.detach()[start_prune_mask]
                else:
                    sugar._scales[...] = nerfmodel.gaussians._scaling.detach()
                    sugar._quaternions[...] = nerfmodel.gaussians._rotation.detach()
                    sugar.all_densities[...] = nerfmodel.gaussians._opacity.detach()
                    sugar._sh_coordinates_dc[...] = nerfmodel.gaussians._features_dc.detach()
                    sugar._sh_coordinates_rest[...] = nerfmodel.gaussians._features_rest.detach()
            
            # Save initialization cache for future runs
            CONSOLE.print(f"💾 Saving initialization cache to {cache_file}...")
            cache_data = {
                'points': points.detach().cpu(),
                'colors': colors.detach().cpu(),
                'sh_levels': sh_levels,
                'sugar_state': {
                    'scales': sugar._scales.detach().cpu(),
                    'quaternions': sugar._quaternions.detach().cpu(),
                    'opacities': sugar.all_densities.detach().cpu(),
                    'sh_dc': sugar._sh_coordinates_dc.detach().cpu(),
                    'sh_rest': sugar._sh_coordinates_rest.detach().cpu(),
                }
            }
            
            # Cache KNN data if it was computed (expensive operation worth caching)
            if sugar.keep_track_of_knn:
                CONSOLE.print("Caching KNN data for future runs...")
                cache_data['sugar_state']['knn_dists'] = sugar.knn_dists.detach().cpu()
                cache_data['sugar_state']['knn_idx'] = sugar.knn_idx.detach().cpu()
            
            torch.save(cache_data, cache_file)
            CONSOLE.print("✓ Initialization cache saved (including KNN data). Future runs will be much faster!")
        
    # Create initialization summary table
    CONSOLE.print()
    CONSOLE.print("═" * 80)
    init_table = Table(title="🎯 SuGaR Model Initialized", show_header=True, header_style="bold magenta")
    init_table.add_column("Property", style="cyan", width=30)
    init_table.add_column("Value", style="green", width=45)
    
    total_params = sum(p.numel() for p in sugar.parameters() if p.requires_grad)
    trainable_params = sum(p.numel() for p in sugar.parameters() if p.requires_grad)
    total_all_params = sum(p.numel() for p in sugar.parameters())
    
    # Calculate memory footprint (rough estimate)
    param_memory_mb = (trainable_params * 4) / (1024 * 1024)  # 4 bytes per float32
    
    init_table.add_row("Total Points (Gaussians)", f"{len(sugar.points):,}")
    init_table.add_row("Trainable Parameters", f"{trainable_params:,}")
    init_table.add_row("Total Parameters", f"{total_all_params:,}")
    init_table.add_row("Estimated Param Memory", f"{param_memory_mb:.0f} MB")
    init_table.add_row("", "")  # Spacer
    init_table.add_row("Checkpoint Output Path", sugar_checkpoint_path)
    
    CONSOLE.print(init_table)
    CONSOLE.print("═" * 80)
    
    # Create parameter shape table (condensed)
    param_table = Table(title="📊 Model Architecture", show_header=True, header_style="bold blue")
    param_table.add_column("Parameter", style="cyan", width=22)
    param_table.add_column("Shape", style="yellow", width=20)
    param_table.add_column("Elements", style="magenta", width=12, justify="right")
    param_table.add_column("Memory", style="green", width=10, justify="right")
    param_table.add_column("Train", style="blue", width=5, justify="center")
    
    for name, param in sugar.named_parameters():
        num_elements = param.numel()
        memory_mb = (num_elements * 4) / (1024 * 1024)  # 4 bytes per float32
        
        # Format memory: KB for small, MB for large
        if memory_mb < 1:
            memory_str = f"{memory_mb * 1024:.0f} KB"
        else:
            memory_str = f"{memory_mb:.1f} MB"
        
        param_table.add_row(
            name, 
            str(list(param.shape)), 
            f"{num_elements:,}",
            memory_str,
            "✓" if param.requires_grad else "✗"
        )
    
    CONSOLE.print(param_table)
    CONSOLE.print("💡 [dim]Monitor full statistics & training history: [cyan]http://localhost:6007[/cyan][/dim]")
 
    torch.cuda.empty_cache()
    
    # Compute scene extent
    t_extent_start = time.time()
    cameras_spatial_extent = sugar.get_cameras_spatial_extent()
    CONSOLE.print(f"⏱️  Scene extent computed in {time.time() - t_extent_start:.2f}s: {cameras_spatial_extent:.4f}")
    
    
    # ====================Initialize optimizer====================
    t_opt_start = time.time()
    if spatial_lr_scale is None:
        spatial_lr_scale = cameras_spatial_extent
        print("Using camera spatial extent as spatial_lr_scale:", spatial_lr_scale)
    
    opt_params = OptimizationParams(
        iterations=num_iterations,
        position_lr_init=position_lr_init,
        position_lr_final=position_lr_final,
        position_lr_delay_mult=position_lr_delay_mult,
        position_lr_max_steps=position_lr_max_steps,
        feature_lr=feature_lr,
        opacity_lr=opacity_lr,
        scaling_lr=scaling_lr,
        rotation_lr=rotation_lr,
    )
    optimizer = SuGaROptimizer(sugar, opt_params, spatial_lr_scale=spatial_lr_scale)
    CONSOLE.print(f"⏱️  Optimizer initialized in {time.time() - t_opt_start:.2f}s")
    
    # Create optimizer table
    CONSOLE.print()
    opt_table = Table(title="⚙️  Optimization Settings", show_header=True, header_style="bold yellow")
    opt_table.add_column("Parameter Group", style="cyan", width=30)
    opt_table.add_column("Learning Rate", style="green", width=15, justify="right")
    opt_table.add_column("Schedule", style="dim", width=30)
    
    # Add spatial LR scale info first
    opt_table.add_row(
        "[bold]Spatial LR Scale[/bold]",
        f"[bold]{spatial_lr_scale:.4f}[/bold]",
        "[dim](based on scene extent)[/dim]"
    )
    opt_table.add_row("", "", "")  # Spacer
    
    for param_group in optimizer.optimizer.param_groups:
        # Determine if this has a schedule
        if param_group['name'] == 'points':
            schedule_info = f"[dim]→ {position_lr_final:.6f} (exponential)[/dim]"
        else:
            schedule_info = "[dim]constant[/dim]"
        
        opt_table.add_row(
            param_group['name'], 
            f"{param_group['lr']:.6f}",
            schedule_info
        )
    
    CONSOLE.print(opt_table)
    CONSOLE.print("═" * 80)
        
        
    # ====================Initialize densifier====================
    t_densify_start = time.time()
    gaussian_densifier = SuGaRDensifier(
        sugar_model=sugar,
        sugar_optimizer=optimizer,
        max_grad=densify_grad_threshold,
        min_opacity=prune_opacity_threshold,
        max_screen_size=densify_screen_size_threshold,
        scene_extent=cameras_spatial_extent,
        percent_dense=densification_percent_distinction,
        )
    CONSOLE.print(f"⏱️  Densifier initialized in {time.time() - t_densify_start:.2f}s")
        
    
    # ====================Loss function====================
    if loss_function == 'l1':
        loss_fn = l1_loss
    elif loss_function == 'l2':
        loss_fn = l2_loss
    elif loss_function == 'l1+dssim':
        def loss_fn(pred_rgb, gt_rgb):
            return (1.0 - dssim_factor) * l1_loss(pred_rgb, gt_rgb) + dssim_factor * (1.0 - ssim(pred_rgb, gt_rgb))
    CONSOLE.print(f'Using loss function: {loss_function}')
    
    
    # ====================Resume from checkpoint if provided====================
    if is_resuming:
        CONSOLE.print(f"\n🔄 [bold yellow]Loading checkpoint state:[/bold yellow] {resume_checkpoint_path}")
        # checkpoint already loaded earlier, now on GPU
        # weights_only=False is safe here since we created these checkpoints ourselves
        checkpoint_gpu = torch.load(resume_checkpoint_path, map_location='cuda', weights_only=False)
        
        # Load model state
        sugar.load_state_dict(checkpoint_gpu['state_dict'])
        CONSOLE.print("✓ Model state loaded")
        
        # Load optimizer state  
        optimizer.load_state_dict(checkpoint_gpu['optimizer_state_dict'])
        CONSOLE.print("✓ Optimizer state loaded")
        
        # Load training progress
        epoch = checkpoint_gpu.get('epoch', 0)
        iteration = checkpoint_gpu.get('iteration', 0)
        train_losses = checkpoint_gpu.get('train_losses', [])
        CONSOLE.print(f"✓ Resuming from epoch {epoch}, iteration {iteration}")
        
        # Free checkpoint memory
        del checkpoint_gpu
        torch.cuda.empty_cache()
        
        # Note: Training will continue from iteration+1
        CONSOLE.print(f"[bold green]✅ Resume complete! Training continues from iteration {iteration + 1}[/bold green]\n")
    else:
        if resume_checkpoint_path:
            CONSOLE.print(f"[bold red]Warning:[/bold red] Resume checkpoint not found: {resume_checkpoint_path}")
            CONSOLE.print("[yellow]Starting from scratch...[/yellow]\n")
        
        epoch = 0
        iteration = 0
        train_losses = []
        
        if initialize_from_trained_3dgs:
            # Start from the 3DGS checkpoint iteration (e.g., if trained to 60K, start SuGaR at 60K)
            # This respects the actual training progress of the input checkpoint
            iteration = iteration_to_load - 1
            CONSOLE.print(f"\n[cyan]ℹ️  Starting from iteration {iteration_to_load} (matches 3DGS checkpoint)[/cyan]")
            CONSOLE.print("[dim]   SuGaR assumes 3DGS model is pre-trained and skips early densification.[/dim]")
            CONSOLE.print(f"[dim]   Training will run: {iteration_to_load} → {num_iterations} = {num_iterations - iteration_to_load} iterations[/dim]\n")
    
    # ====================Start training====================
    sugar.train()
    start_iteration = iteration + 1  # Track starting point for accurate speed/ETA
    t0 = time.time()
    loss = None  # Initialize to handle case where no training iterations run (e.g., resume at final iteration)
    
    # Track which training phase we've announced
    announced_densification = False
    announced_stable = False
    
    # Print training header with expectations
    CONSOLE.print("")
    CONSOLE.print("[bold cyan]═" * 80 + "[/bold cyan]")
    
    # Calculate mid-point iteration between start and end
    mid_iteration = (start_iteration + num_iterations) // 2
    
    training_info_panel = Panel(
        f"[bold white]Total Iterations:[/bold white] [cyan]{num_iterations:,}[/cyan]\n"
        f"[bold white]Starting From:[/bold white] [cyan]{start_iteration:,}[/cyan]\n"
        f"[bold white]Actual Training:[/bold white] [cyan]{num_iterations - start_iteration + 1:,} iterations[/cyan]\n"
        "\n"
        "[bold yellow]📈 Expected Loss Progression:[/bold yellow]\n"
        f"  • Start (iter {start_iteration:,}): ~0.17\n"
        f"  • Mid (iter {mid_iteration:,}): ~0.10\n"
        f"  • End (iter {num_iterations:,}): ~0.05\n"
        "\n"
        "[bold green]✓ Checkpoints:[/bold green]\n"
        f"  • Auto-save every: {save_model_every_n_iterations:,} iterations\n"
        f"  • Milestones: {', '.join(str(x) for x in save_milestones)}\n"
        "\n"
        "[bold blue]📊 Evaluation:[/bold blue]\n"
        f"  • Test iterations: {', '.join(str(x) for x in test_iterations)}\n"
        "\n"
        "[dim]Monitor progress in TensorBoard: http://localhost:6007[/dim]",
        title="🚀 Training Configuration",
        border_style="cyan",
        expand=False
    )
    CONSOLE.print(training_info_panel)
    
    CONSOLE.print("")
    CONSOLE.print("[bold cyan]═" * 80 + "[/bold cyan]")
    CONSOLE.print("[bold cyan]" + " " * 30 + "TRAINING STARTED" + " " * 34 + "[/bold cyan]")
    CONSOLE.print("[bold cyan]═" * 80 + "[/bold cyan]")
    CONSOLE.print("")
    
    # Initialize speed tracking for accurate it/s calculation
    speed_window_size = 100  # Calculate speed over last 100 iterations
    speed_window_times = []
    speed_window_iters = []
    
    # Initialize Rich Progress bar for training
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=10),
        MofNCompleteColumn(),
        TextColumn("│"),
        TextColumn("[cyan]L:{task.fields[loss]:.3f}[/cyan]"),
        TextColumn("│"),
        TextColumn("[yellow]V:{task.fields[vram]:.0f}%[/yellow]"),
        TextColumn("[magenta]R:{task.fields[ram]:.0f}%[/magenta]"),
        TextColumn("[green]C:{task.fields[cpu]:.0f}%[/green]"),
        TextColumn("│"),
        TextColumn("{task.fields[speed]:.1f}it/s"),
        TextColumn("│"),
        TimeElapsedColumn(),
        TextColumn("│"),
        TimeRemainingColumn(),
        console=CONSOLE,
    )
    
    try:
        progress.start()
        training_task = progress.add_task(
            "[cyan]Coarse Training",
            total=num_iterations - start_iteration,
            loss=0.0,
            vram=0.0,
            ram=0.0,
            cpu=0.0,
            speed=0.0
        )
        
        for batch in range(9_999_999):
            # print(f"DEBUG: MAIN BATCH LOOP - batch {batch}")
            if iteration >= num_iterations:
                break
            
            # Shuffle images
            shuffled_idx = torch.randperm(len(nerfmodel.training_cameras))
            train_num_images = len(shuffled_idx)
            
            # We iterate on images
            for i in range(0, train_num_images, train_num_images_per_batch):
                # print(f"DEBUG: IMAGE LOOP - processing images {i} to {i+train_num_images_per_batch}")
                iteration += 1

                # Announce training phase transitions(similar to mip-splatting)
                if iteration == densify_from_iter and num_iterations >= densify_from_iter and not announced_densification:
                    if num_iterations < densify_until_iter:
                        # Training ends before densification finishes
                        CONSOLE.print(f"\n[bold yellow]📈 Densification Phase[/bold yellow] (iters {densify_from_iter}-{num_iterations})")
                        CONSOLE.print(f"  [dim]→ Training ends before densification completes (default: until {densify_until_iter})[/dim]")
                    else:
                        # Full densification phase
                        CONSOLE.print(f"\n[bold yellow]📈 Densification Phase[/bold yellow] (iters {densify_from_iter}-{densify_until_iter})")
                    CONSOLE.print(f"  [dim]→ Split/clone/prune every {densification_interval} iterations[/dim]")
                    announced_densification = True
                elif iteration == densify_until_iter and num_iterations >= densify_until_iter and not announced_stable:
                    CONSOLE.print(f"\n[bold green]🎯 Stable Training Phase[/bold green] (iters {densify_until_iter}-{num_iterations})")
                    CONSOLE.print(f"  [dim]→ No more densification, focusing on refinement[/dim]")
                    announced_stable = True
                
                # Update learning rates
                optimizer.update_learning_rate(iteration)
                
                # Prune low-opacity gaussians for optimizing triangles
                if (
                    regularize and prune_when_starting_regularization and iteration == regularize_from + 1
                    ) or (
                    (iteration-1) in prune_low_opacity_gaussians_at
                    ):
                    CONSOLE.print("🔧 [dim]Pruning low-opacity Gaussians...[/dim]")
                    prune_mask = (gaussian_densifier.model.strengths < prune_hard_opacity_threshold).squeeze()
                    gaussian_densifier.prune_points(prune_mask)
                    CONSOLE.print(f"🔧 [green]Pruned → {sugar.n_points:,} Gaussians[/green]")
                    if regularize and iteration >= start_reset_neighbors_from:
                        sugar.reset_neighbors()
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                
                start_idx = i
                end_idx = min(i+train_num_images_per_batch, train_num_images)
                
                camera_indices = shuffled_idx[start_idx:end_idx]
                
                # Computing rgb predictions
                if not no_rendering:
                    # Wrap rendering in gradient checkpointing if enabled (Phase 2 optimization)
                    if use_gradient_checkpointing:
                        # Create wrapper function for checkpointing
                        def render_fn():
                            return sugar.render_image_gaussian_rasterizer( 
                                camera_indices=camera_indices.item(),
                                verbose=False,
                                bg_color = bg_tensor,
                                sh_deg=current_sh_levels-1,
                                sh_rotations=None,
                                compute_color_in_rasterizer=compute_color_in_rasterizer,
                                compute_covariance_in_rasterizer=True,
                                return_2d_radii=True,
                                quaternions=None,
                                use_same_scale_in_all_directions=use_same_scale_in_all_directions,
                                return_opacities=enforce_entropy_regularization,
                            )
                        outputs = torch.utils.checkpoint.checkpoint(render_fn, use_reentrant=False)
                    else:
                        outputs = sugar.render_image_gaussian_rasterizer( 
                            camera_indices=camera_indices.item(),
                            verbose=False,
                            bg_color = bg_tensor,
                            sh_deg=current_sh_levels-1,
                            sh_rotations=None,
                            compute_color_in_rasterizer=compute_color_in_rasterizer,
                            compute_covariance_in_rasterizer=True,
                            return_2d_radii=True,
                            quaternions=None,
                            use_same_scale_in_all_directions=use_same_scale_in_all_directions,
                            return_opacities=enforce_entropy_regularization,
                        )
                    pred_rgb = outputs['image'].view(-1, 
                        sugar.image_height, 
                        sugar.image_width, 
                        3)
                    radii = outputs['radii']
                    viewspace_points = outputs['viewspace_points']
                    if enforce_entropy_regularization:
                        opacities = outputs['opacities']
                    
                    pred_rgb = pred_rgb.transpose(-1, -2).transpose(-2, -3)  # TODO: Change for torch.permute
                    
                    # Gather rgb ground truth
                    gt_image = nerfmodel.get_gt_image(camera_indices=camera_indices)           
                    gt_rgb = gt_image.view(-1, sugar.image_height, sugar.image_width, 3)
                    gt_rgb = gt_rgb.transpose(-1, -2).transpose(-2, -3)
                        
                    # Compute loss 
                    loss = loss_fn(pred_rgb, gt_rgb)
                    
                    # Free intermediate tensors immediately (VRAM optimization)
                    del outputs, gt_image
                    if not enforce_entropy_regularization:
                        del radii, viewspace_points
                    
                    # Track loss components for TensorBoard (zero VRAM cost)
                    loss_components = {'rendering': loss.item()}
                            
                    if enforce_entropy_regularization and iteration > start_entropy_regularization_from and iteration < end_entropy_regularization_at:
                        if iteration == start_entropy_regularization_from + 1:
                            CONSOLE.print("🎯 [cyan]Starting entropy regularization[/cyan]")
                        if iteration == end_entropy_regularization_at - 1:
                            CONSOLE.print("🎯 [cyan]Stopping entropy regularization[/cyan]")
                        visibility_filter = radii > 0
                        if visibility_filter is not None:
                            vis_opacities = opacities[visibility_filter]
                        else:
                            vis_opacities = opacities
                        entropy_loss = entropy_regularization_factor * (
                            - vis_opacities * torch.log(vis_opacities + 1e-10)
                            - (1 - vis_opacities) * torch.log(1 - vis_opacities + 1e-10)
                            ).mean()
                        loss = loss + entropy_loss
                        loss_components['entropy_regularization'] = entropy_loss.item()
                        
                        # Clean up intermediate tensors (VRAM optimization)
                        del visibility_filter, vis_opacities, entropy_loss
                        
                    # Depth-Normal consistency
                    if enforce_depth_normal_consistency and iteration > start_dn_consistency_from:
                        if iteration == start_dn_consistency_from + 1:
                            use_full_res = args.full_res_normals if hasattr(args, 'full_res_normals') else False
                            res_mode = "full" if use_full_res else "half"
                            checkpoint_mode = "checkpointed" if use_gradient_checkpointing else "standard"
                            
                            if use_full_res and use_gradient_checkpointing:
                                CONSOLE.print(f"🎨 [magenta]Starting depth-normal consistency[/magenta] [dim](full-res, {checkpoint_mode}, max quality)[/dim]")
                            elif not use_full_res and use_gradient_checkpointing:
                                CONSOLE.print(f"🎨 [magenta]Starting depth-normal consistency[/magenta] [dim](half-res, cannot checkpoint, ~1-3% quality trade-off for VRAM)[/dim]")
                            else:
                                CONSOLE.print(f"🎨 [magenta]Starting depth-normal consistency[/magenta] [dim]({res_mode} resolution, {checkpoint_mode})[/dim]")
                        
                        # Save original resolution
                        original_h, original_w = sugar.image_height, sugar.image_width
                        
                        # Downscale if not using full resolution (default behavior for VRAM savings)
                        use_full_res_normals = args.full_res_normals if hasattr(args, 'full_res_normals') else False
                        if not use_full_res_normals:
                            sugar.image_height = original_h // 2
                            sugar.image_width = original_w // 2
                        
                        # Render depth and normal with gradient checkpointing (if using full-res)
                        # NOTE: Cannot checkpoint with half-res - modifies sugar.image_height/width (global state)
                        if use_gradient_checkpointing and use_full_res_normals:
                            def render_depth_normal_fn():
                                return sugar.render_depth_and_normal(camera_indices=camera_indices.item())
                            depth_img, normal_img = torch.utils.checkpoint.checkpoint(render_depth_normal_fn, use_reentrant=False)
                        else:
                            depth_img, normal_img = sugar.render_depth_and_normal(camera_indices=camera_indices.item())
                        
                        # Upsample to full resolution if we downscaled
                        if not use_full_res_normals:
                            # Upsample depth using nearest neighbor
                            depth_img = torch.nn.functional.interpolate(
                                depth_img.unsqueeze(0).unsqueeze(0), 
                                size=(original_h, original_w), 
                                mode='nearest'
                            ).squeeze()
                            
                            # Upsample normal using bilinear interpolation
                            normal_img = torch.nn.functional.interpolate(
                                normal_img.permute(2,0,1).unsqueeze(0),
                                size=(original_h, original_w),
                                mode='bilinear',
                                align_corners=False
                            ).squeeze(0).permute(1,2,0)
                            
                            # Restore original resolution
                            sugar.image_height = original_h
                            sugar.image_width = original_w
                        
                        # Compute loss (same as before)
                        normal_error = depth_normal_consistency_loss(
                            depth=depth_img[None],  # Shape is (1, height, width) 
                            normal=normal_img.permute(2, 0, 1),  # Shape is (3, height, width)
                            camera=nerfmodel.training_cameras.gs_cameras[camera_indices.item()],
                            scale_rendered_normals=False,
                            return_normal_maps=False,
                        )
                        dn_loss_scaled = dn_consistency_factor * normal_error
                        loss = loss + dn_loss_scaled
                        loss_components['depth_normal_consistency'] = dn_loss_scaled.item()
                        
                        # Clean up intermediate tensors (VRAM optimization)
                        del depth_img, normal_img, normal_error, dn_loss_scaled
                    
                    # SuGaR regularization
                    if regularize:
                        if iteration == regularize_from:
                            CONSOLE.print("Starting regularization...")
                            # sugar.reset_neighbors()
                        if iteration > regularize_from:
                            visibility_filter = radii > 0
                            # Skip reset_neighbors on first iteration after resume (neighbors already exist from checkpoint)
                            if (iteration >= start_reset_neighbors_from) and ((iteration == regularize_from + 1) or (iteration % reset_neighbors_every == 0)) and (iteration != start_iteration):
                                CONSOLE.print("🔄 [dim]Resetting neighbors...[/dim]")
                                sugar.reset_neighbors()
                                torch.cuda.empty_cache()  # Clear CUDA cache after neighbor reset
                                import gc
                                gc.collect()  # Force Python garbage collection
                            neighbor_idx = sugar.get_neighbors_of_random_points(num_samples=regularity_samples,)
                            # Don't apply visibility_filter when using random sampling (already samples randomly)
                            # Only apply if using ALL points (regularity_samples == -1)
                            if visibility_filter is not None and regularity_samples == -1:
                                neighbor_idx = neighbor_idx[visibility_filter]
    
                            # Run SDF sampling every 5th iteration to save 3-4GB VRAM (minimal quality impact)
                            if regularize_sdf and iteration > start_sdf_regularization_from and iteration % 5 == 0:
                                if iteration == start_sdf_regularization_from + 1:
                                    CONSOLE.print("ℹ️  [dim]Starting SDF regularization (every 5th iteration for VRAM savings)[/dim]")
                                
                                sampling_mask = visibility_filter
                                
                                if (use_sdf_estimation_loss or enforce_samples_to_be_on_surface) and iteration > start_sdf_estimation_from:
                                    if iteration == start_sdf_estimation_from + 1:
                                        CONSOLE.print("ℹ️  [dim]Starting SDF estimation loss[/dim]")
                                    fov_camera = nerfmodel.training_cameras.p3d_cameras[camera_indices.item()]
                                    
                                    if use_projection_as_estimation:
                                        pass
                                    else:
                                        # Render a depth map using gaussian splatting
                                        if backpropagate_gradients_through_depth:                                
                                            point_depth = fov_camera.get_world_to_view_transform().transform_points(sugar.points)[..., 2:].expand(-1, 3)
                                            max_depth = point_depth.max()
                                            depth = sugar.render_image_gaussian_rasterizer(
                                                        camera_indices=camera_indices.item(),
                                                        bg_color=max_depth + torch.zeros(3, dtype=torch.float, device=sugar.device),
                                                        sh_deg=0,
                                                        compute_color_in_rasterizer=False,#compute_color_in_rasterizer,
                                                        compute_covariance_in_rasterizer=True,
                                                        return_2d_radii=False,
                                                        use_same_scale_in_all_directions=False,
                                                        point_colors=point_depth,
                                                    )[..., 0]
                                        else:
                                            with torch.no_grad():
                                                point_depth = fov_camera.get_world_to_view_transform().transform_points(sugar.points)[..., 2:].expand(-1, 3)
                                                max_depth = point_depth.max()
                                                depth = sugar.render_image_gaussian_rasterizer(
                                                            camera_indices=camera_indices.item(),
                                                            bg_color=max_depth + torch.zeros(3, dtype=torch.float, device=sugar.device),
                                                            sh_deg=0,
                                                            compute_color_in_rasterizer=False,#compute_color_in_rasterizer,
                                                            compute_covariance_in_rasterizer=True,
                                                            return_2d_radii=False,
                                                            use_same_scale_in_all_directions=False,
                                                            point_colors=point_depth,
                                                        )[..., 0]
                                    
                                    # If needed, compute which gaussians are close to the surface in the depth map.
                                    # Then, we sample points only in these gaussians.
                                    # TODO: Compute projections only for gaussians in visibility filter.
                                    # TODO: Is the torch.no_grad() a good idea?
                                    if sample_only_in_gaussians_close_to_surface:
                                        with torch.no_grad():
                                            gaussian_to_camera = torch.nn.functional.normalize(fov_camera.get_camera_center() - sugar.points, dim=-1)
                                            gaussian_centers_in_camera_space = fov_camera.get_world_to_view_transform().transform_points(sugar.points)
                                            
                                            gaussian_centers_z = gaussian_centers_in_camera_space[..., 2] + 0.
                                            gaussian_centers_map_z = sugar.get_points_depth_in_depth_map(fov_camera, depth, gaussian_centers_in_camera_space)
                                            
                                            gaussian_standard_deviations = (
                                                sugar.scaling * quaternion_apply(quaternion_invert(sugar.quaternions), gaussian_to_camera)
                                                ).norm(dim=-1)
                                        
                                            gaussians_close_to_surface = (gaussian_centers_map_z - gaussian_centers_z).abs() < close_gaussian_threshold * gaussian_standard_deviations
                                            sampling_mask = sampling_mask * gaussians_close_to_surface
                                
                                n_gaussians_in_sampling = sampling_mask.sum()
                                if n_gaussians_in_sampling > 0:
                                    sdf_samples, sdf_gaussian_idx = sugar.sample_points_in_gaussians(
                                        num_samples=n_samples_for_sdf_regularization, 
                                        sampling_scale_factor=sdf_sampling_scale_factor,
                                        mask=sampling_mask,
                                        probabilities_proportional_to_volume=sdf_sampling_proportional_to_volume,
                                        )
                                    
                                    if use_sdf_estimation_loss or use_sdf_better_normal_loss:
                                        # Compute SDF fields with optional gradient checkpointing
                                        if use_gradient_checkpointing:
                                            def get_field_values_fn():
                                                return sugar.get_field_values(
                                                    sdf_samples, sdf_gaussian_idx, 
                                                    return_sdf=(use_sdf_estimation_loss or enforce_samples_to_be_on_surface) and (sdf_estimation_mode=='sdf') and iteration > start_sdf_estimation_from, 
                                                    density_threshold=density_threshold, density_factor=density_factor, 
                                                    return_sdf_grad=False, sdf_grad_max_value=10.,
                                                    return_closest_gaussian_opacities=use_sdf_better_normal_loss and iteration > start_sdf_better_normal_from,
                                                    return_beta=(use_sdf_estimation_loss or enforce_samples_to_be_on_surface) and (sdf_estimation_mode=='density') and iteration > start_sdf_estimation_from,
                                                )
                                            fields = torch.utils.checkpoint.checkpoint(get_field_values_fn, use_reentrant=False)
                                        else:
                                            fields = sugar.get_field_values(
                                                sdf_samples, sdf_gaussian_idx, 
                                                return_sdf=(use_sdf_estimation_loss or enforce_samples_to_be_on_surface) and (sdf_estimation_mode=='sdf') and iteration > start_sdf_estimation_from, 
                                                density_threshold=density_threshold, density_factor=density_factor, 
                                                return_sdf_grad=False, sdf_grad_max_value=10.,
                                                return_closest_gaussian_opacities=use_sdf_better_normal_loss and iteration > start_sdf_better_normal_from,
                                                return_beta=(use_sdf_estimation_loss or enforce_samples_to_be_on_surface) and (sdf_estimation_mode=='density') and iteration > start_sdf_estimation_from,
                                            )
                                    
                                    if (use_sdf_estimation_loss or enforce_samples_to_be_on_surface) and iteration > start_sdf_estimation_from:
                                        # Compute the depth of the points in the gaussians
                                        if use_projection_as_estimation:        
                                            proj_mask = torch.ones_like(sdf_samples[..., 0], dtype=torch.bool)
                                            samples_gaussian_normals = sugar.get_normals(estimate_from_points=False)[sdf_gaussian_idx]
                                            sdf_estimation = ((sdf_samples - sugar.points[sdf_gaussian_idx]) * samples_gaussian_normals).sum(dim=-1)  # Shape is (n_samples,)
                                        else:
                                            sdf_samples_in_camera_space = fov_camera.get_world_to_view_transform().transform_points(sdf_samples)
                                            sdf_samples_z = sdf_samples_in_camera_space[..., 2] + 0.
                                            proj_mask = sdf_samples_z > fov_camera.znear
                                            sdf_samples_map_z = sugar.get_points_depth_in_depth_map(fov_camera, depth, sdf_samples_in_camera_space[proj_mask])
                                            sdf_estimation = sdf_samples_map_z - sdf_samples_z[proj_mask]
                                        
                                        if not sample_only_in_gaussians_close_to_surface:
                                            if normalize_by_sdf_std:
                                                print("Setting normalize_by_sdf_std to False because sample_only_in_gaussians_close_to_surface is False.")
                                                normalize_by_sdf_std = False
                                        
                                        with torch.no_grad():
                                            if normalize_by_sdf_std:
                                                sdf_sample_std = gaussian_standard_deviations[sdf_gaussian_idx][proj_mask]
                                            else:
                                                sdf_sample_std = sugar.get_cameras_spatial_extent() / 10.
                                        
                                        if use_sdf_estimation_loss:
                                            if sdf_estimation_mode == 'sdf':
                                                sdf_values = fields['sdf'][proj_mask]
                                                if squared_sdf_estimation_loss:
                                                    sdf_estimation_loss = ((sdf_values - sdf_estimation.abs()) / sdf_sample_std).pow(2)
                                                else:
                                                    sdf_estimation_loss = (sdf_values - sdf_estimation.abs()).abs() / sdf_sample_std
                                                sdf_loss_scaled = sdf_estimation_factor * sdf_estimation_loss.clamp(max=10.*sugar.get_cameras_spatial_extent()).mean()
                                                loss = loss + sdf_loss_scaled
                                                loss_components['sdf_estimation'] = sdf_loss_scaled.item()
                                            elif sdf_estimation_mode == 'density':
                                                beta = fields['beta'][proj_mask]
                                                densities = fields['density'][proj_mask]
                                                target_densities = torch.exp(-0.5 * sdf_estimation.pow(2) / beta.pow(2))
                                                if squared_sdf_estimation_loss:
                                                    sdf_estimation_loss = ((densities - target_densities)).pow(2)
                                                else:
                                                    sdf_estimation_loss = (densities - target_densities).abs()
                                                sdf_loss_scaled = sdf_estimation_factor * sdf_estimation_loss.mean()
                                                loss = loss + sdf_loss_scaled
                                                loss_components['sdf_density'] = sdf_loss_scaled.item()
                                            else:
                                                raise ValueError(f"Unknown sdf_estimation_mode: {sdf_estimation_mode}")
    
                                        if enforce_samples_to_be_on_surface:
                                            if squared_samples_on_surface_loss:
                                                samples_on_surface_loss = (sdf_estimation / sdf_sample_std).pow(2)
                                            else:
                                                samples_on_surface_loss = sdf_estimation.abs() / sdf_sample_std
                                            loss = loss + samples_on_surface_factor * samples_on_surface_loss.clamp(max=10.*sugar.get_cameras_spatial_extent()).mean()
                                            
                                    if use_sdf_better_normal_loss and (iteration > start_sdf_better_normal_from):
                                        if iteration == start_sdf_better_normal_from + 1:
                                            CONSOLE.print("\n---INFO---\nStarting SDF better normal loss.")
                                        closest_gaussians_idx = sugar.knn_idx[sdf_gaussian_idx]
                                        # Compute minimum scaling
                                        closest_min_scaling = sugar.scaling.min(dim=-1)[0][closest_gaussians_idx].detach().view(len(sdf_samples), -1)
                                        
                                        # Compute normals and flip their sign if needed
                                        closest_gaussian_normals = sugar.get_normals(estimate_from_points=False)[closest_gaussians_idx]
                                        samples_gaussian_normals = sugar.get_normals(estimate_from_points=False)[sdf_gaussian_idx]
                                        closest_gaussian_normals = closest_gaussian_normals * torch.sign(
                                            (closest_gaussian_normals * samples_gaussian_normals[:, None]).sum(dim=-1, keepdim=True)
                                            ).detach()
                                        
                                        # Compute weights for normal regularization, based on the gradient of the sdf
                                        closest_gaussian_opacities = fields['closest_gaussian_opacities'].detach()  # Shape is (n_samples, n_neighbors)
                                        normal_weights = ((sdf_samples[:, None] - sugar.points[closest_gaussians_idx]) * closest_gaussian_normals).sum(dim=-1).abs()  # Shape is (n_samples, n_neighbors)
                                        if sdf_better_normal_gradient_through_normal_only:
                                            normal_weights = normal_weights.detach()
                                        normal_weights =  closest_gaussian_opacities * normal_weights / closest_min_scaling.clamp(min=1e-6)**2  # Shape is (n_samples, n_neighbors)
                                        
                                        # The weights should have a sum of 1 because of the eikonal constraint
                                        normal_weights_sum = normal_weights.sum(dim=-1).detach()  # Shape is (n_samples,)
                                        normal_weights = normal_weights / normal_weights_sum.unsqueeze(-1).clamp(min=1e-6)  # Shape is (n_samples, n_neighbors)
                                        
                                        # Compute regularization loss
                                        sdf_better_normal_loss = (samples_gaussian_normals - (normal_weights[..., None] * closest_gaussian_normals).sum(dim=-2)
                                                                  ).pow(2).sum(dim=-1)  # Shape is (n_samples,)
                                        loss = loss + sdf_better_normal_factor * sdf_better_normal_loss.mean()
                                else:
                                    CONSOLE.log("WARNING: No gaussians available for sampling.")
                                    
                else:
                    loss = 0.
                    
                # Surface mesh optimization
                if bind_to_surface_mesh:
                    surface_mesh = sugar.surface_mesh
                    
                    if use_surface_mesh_laplacian_smoothing_loss:
                        loss = loss + surface_mesh_laplacian_smoothing_factor * mesh_laplacian_smoothing(
                            surface_mesh, method=surface_mesh_laplacian_smoothing_method)
                    
                    if use_surface_mesh_normal_consistency_loss:
                        loss = loss + surface_mesh_normal_consistency_factor * mesh_normal_consistency(surface_mesh)
                
                # Update parameters
                loss.backward()
                
                # Densification
                with torch.no_grad():
                    if (not no_rendering) and (iteration < densify_until_iter):
                        gaussian_densifier.update_densification_stats(viewspace_points, radii, visibility_filter=radii>0)
    
                        if iteration > densify_from_iter and iteration % densification_interval == 0:
                            size_threshold = gaussian_densifier.max_screen_size if iteration > opacity_reset_interval else None
                            gaussian_densifier.densify_and_prune(densify_grad_threshold, prune_opacity_threshold, 
                                                        cameras_spatial_extent, size_threshold)
                            CONSOLE.print(f"🔧 [dim]Densified & pruned → {len(sugar.points):,} Gaussians[/dim]")
                            
                            if regularize and (iteration > regularize_from) and (iteration >= start_reset_neighbors_from):
                                sugar.reset_neighbors()
                        
                        if iteration % opacity_reset_interval == 0:
                            gaussian_densifier.reset_opacity()
                            CONSOLE.print("🔧 [dim]Opacity reset[/dim]")
                
                # Optimization step
                optimizer.step()
                optimizer.zero_grad(set_to_none = True)
                
                # Aggressive memory management to prevent VRAM fragmentation and spilling
                # Clear EVERY iteration when starting above 13GB baseline (prevents 16GB spill)
                torch.cuda.empty_cache()
                
                # Additional cleanup every 25 iterations
                if iteration % 25 == 0:
                    # Clear image cache - only keep last 10 images (saves ~5GB)
                    GSCamera.clear_image_cache()
                    torch.cuda.synchronize()  # Wait for GPU operations to complete
                
                if iteration % 50 == 0:
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    # Tensor memory profiling (if enabled)
                    if hasattr(args, 'profile_tensors') and args.profile_tensors:
                        from sugar_utils.memory_profiler import log_tensor_memory_breakdown
                        log_tensor_memory_breakdown(tb_writer, iteration, interval=50, log_file=tensor_profile_log)
                
                if iteration==1 or iteration % print_loss_every_n_iterations == 0:
                    train_losses.append(loss.detach().item())
                    time_elapsed = (time.time() - t0) / 60.
                    
                    # Get resource usage
                    mem_free, mem_total = torch.cuda.mem_get_info()
                    vram_percent = ((mem_total - mem_free) / mem_total) * 100
                    vram_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
                    vram_reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # Convert to GB
                    
                    ram_info = psutil.virtual_memory()
                    ram_percent = ram_info.percent
                    
                    cpu_percent = psutil.cpu_percent(interval=0)
                    
                    # Calculate ETA (use elapsed iterations, not absolute iteration number)
                    iterations_done = iteration - start_iteration + 1
                    iterations_remaining = num_iterations - iteration
                    
                    # Track speed over rolling window for accurate it/s calculation
                    current_time = time.time()
                    speed_window_times.append(current_time)
                    speed_window_iters.append(iteration)
                    
                    # Keep only last 100 iterations in window
                    if len(speed_window_times) > speed_window_size:
                        speed_window_times.pop(0)
                        speed_window_iters.pop(0)
                    
                    # Calculate speed from window (more accurate than total time)
                    if len(speed_window_times) >= 2:
                        window_time_elapsed = speed_window_times[-1] - speed_window_times[0]
                        window_iters_done = speed_window_iters[-1] - speed_window_iters[0]
                        if window_time_elapsed > 0:
                            its_per_sec = window_iters_done / window_time_elapsed
                        else:
                            its_per_sec = 0
                    else:
                        its_per_sec = 0
                    
                    # Calculate ETA from average time per iteration
                    if iterations_done > 0:
                        avg_time_per_iter = time_elapsed / iterations_done
                        eta_minutes = avg_time_per_iter * iterations_remaining
                    else:
                        eta_minutes = 0
                    
                    # Update Rich progress bar with current metrics
                    progress.update(
                        training_task,
                        completed=iterations_done,
                        loss=loss.item(),
                        vram=vram_percent,
                        ram=ram_percent,
                        cpu=cpu_percent,
                        speed=its_per_sec
                    )
                    
                    # Add text-based progress for log files every 100 iterations (when running with nohup/redirect)
                    if not sys.stdout.isatty() and iteration % 100 == 0:
                        eta_str = f"{int(eta_minutes//60)}h {int(eta_minutes%60)}m" if eta_minutes >= 60 else f"{int(eta_minutes)}m"
                        print(f"[{iteration}/{num_iterations}] Loss: {loss.item():.4f} | "
                              f"VRAM: {vram_percent:.0f}% | Speed: {its_per_sec:.1f}it/s | ETA: {eta_str}")
                    
                    # TensorBoard logging (keep all detailed logging)
                    tb_writer.add_scalar('Loss/total', loss.item(), iteration)
                    
                    # Log loss components breakdown for analysis (zero VRAM cost)
                    for component_name, component_value in loss_components.items():
                        # print(f"DEBUG: Loop through loss components for TensorBoard logging")
                        tb_writer.add_scalar(f'Loss/component_{component_name}', component_value, iteration)
                    
                    # Time and performance metrics
                    tb_writer.add_scalar('Time/minutes_per_iter', time_elapsed, iteration)
                    tb_writer.add_scalar('Performance/iterations_per_sec', its_per_sec, iteration)
                    tb_writer.add_scalar('Performance/eta_minutes', eta_minutes, iteration)
                    
                    # System resource metrics
                    tb_writer.add_scalar('VRAM/allocated_GB', vram_allocated, iteration)
                    tb_writer.add_scalar('VRAM/reserved_GB', vram_reserved, iteration)
                    tb_writer.add_scalar('VRAM/percent', vram_percent, iteration)
                    tb_writer.add_scalar('System/RAM_percent', ram_percent, iteration)
                    tb_writer.add_scalar('System/CPU_percent', cpu_percent, iteration)
                    
                    with torch.no_grad():
                        # TensorBoard parameter statistics (keep all detailed logging)
                        tb_writer.add_scalar('Stats/points_mean', sugar.points.mean().item(), iteration)
                        tb_writer.add_scalar('Stats/points_std', sugar.points.std().item(), iteration)
                        tb_writer.add_scalar('Stats/scales_mean', sugar.scaling.mean().item(), iteration)
                        tb_writer.add_scalar('Stats/scales_std', sugar.scaling.std().item(), iteration)
                        tb_writer.add_histogram('Histograms/scales', sugar.scaling.detach().cpu(), iteration)
                        tb_writer.add_scalar('Stats/quaternions_mean', sugar.quaternions.mean().item(), iteration)
                        tb_writer.add_scalar('Stats/quaternions_std', sugar.quaternions.std().item(), iteration)
                        tb_writer.add_scalar('Stats/sh_dc_mean', sugar._sh_coordinates_dc.mean().item(), iteration)
                        tb_writer.add_scalar('Stats/sh_dc_std', sugar._sh_coordinates_dc.std().item(), iteration)
                        tb_writer.add_scalar('Stats/sh_rest_mean', sugar._sh_coordinates_rest.mean().item(), iteration)
                        tb_writer.add_scalar('Stats/sh_rest_std', sugar._sh_coordinates_rest.std().item(), iteration)
                        tb_writer.add_scalar('Stats/opacities_mean', sugar.strengths.mean().item(), iteration)
                        tb_writer.add_scalar('Stats/opacities_std', sugar.strengths.std().item(), iteration)
                        tb_writer.add_histogram('Histograms/opacities', sugar.strengths.detach().cpu(), iteration)
                        
                        # Number of active Gaussians (opacity > threshold)
                        n_active = (sugar.strengths > 0.01).sum().item()
                        tb_writer.add_scalar('Stats/n_active_gaussians', n_active, iteration)
                        
                        if regularize_sdf and iteration > start_sdf_regularization_from:
                            tb_writer.add_scalar('Stats/n_gaussians_sdf_sampling', n_gaussians_in_sampling, iteration)
                    
                    # Flush TensorBoard to ensure data is written to disk
                    # Flush frequency matches print_loss_every_n_iterations (every 10 iterations)
                    tb_writer.flush()
                    
                # Test evaluation
                if use_eval_split and iteration in test_iterations:
                    # Print newline before evaluation table
                    print()
                    CONSOLE.print(f"  [bold cyan]Evaluation Results - Iteration [cyan]{iteration:>5}[/cyan][/bold cyan]  ")
                    
                    sugar.eval()
                    with torch.no_grad():
                        test_loss = 0.0
                        test_psnr = 0.0
                        test_ssim = 0.0
                        test_lpips = 0.0
                        test_render_time = 0.0
                        
                        # Check if test cameras are available
                        if nerfmodel.test_cameras is None:
                            CONSOLE.print("Warning: No test cameras available. Skipping test evaluation.")
                        else:
                            n_test = len(nerfmodel.test_cameras.gs_cameras)
                            
                            for test_idx in range(n_test):
                                # print(f"DEBUG: Loop through test images for evaluation - image {test_idx}/{n_test}")
                                # Time the rendering
                                render_start = torch.cuda.Event(enable_timing=True)
                                render_end = torch.cuda.Event(enable_timing=True)
                                render_start.record()
                                
                                # Render using test cameras (match training loop format)
                                outputs = sugar.render_image_gaussian_rasterizer(
                                    nerf_cameras=nerfmodel.test_cameras,
                                    camera_indices=test_idx,
                                    bg_color=bg_tensor,
                                    sh_deg=sugar.sh_levels-1,
                                    compute_color_in_rasterizer=True,
                                    compute_covariance_in_rasterizer=True,
                                    return_2d_radii=True,
                                )
                                test_image = outputs['image'].reshape(-1, sugar.image_height, sugar.image_width, 3)
                                test_image = test_image.transpose(-1, -2).transpose(-2, -3)
                                
                                render_end.record()
                                torch.cuda.synchronize()
                                test_render_time += render_start.elapsed_time(render_end)
                                
                                # Get ground truth and format to match rendered image
                                test_gt_raw = nerfmodel.test_cameras.gs_cameras[test_idx].original_image.cuda()
                                test_gt = test_gt_raw.reshape(-1, sugar.image_height, sugar.image_width, 3)
                                test_gt = test_gt.transpose(-1, -2).transpose(-2, -3)
                                
                                test_l1 = l1_loss(test_image, test_gt)
                                test_loss += test_l1.item()
                                
                                # Calculate PSNR
                                # NOTE: PSNR calculation may be incorrect due to tensor format mismatch
                                # Showing ~8-9 dB when it should be ~24-28 dB. Does not affect training.
                                # Loss values are correct. TODO: Fix tensor format/range matching.
                                mse = torch.mean((test_image - test_gt) ** 2)
                                test_psnr += (20 * torch.log10(1.0 / torch.sqrt(mse))).item()
                                
                                # Calculate SSIM and LPIPS on same loaded images (optimized)
                                test_ssim += ssim(test_image, test_gt).mean().item()
                                test_lpips += lpips_fn(test_image, test_gt).mean().item()
                                
                                # Clean up test tensors (VRAM optimization ~50-100MB per image)
                                del outputs, test_image, test_gt, test_gt_raw
                            
                            test_loss /= n_test
                            test_psnr /= n_test
                            test_ssim /= n_test
                            test_lpips /= n_test
                            avg_render_time = test_render_time / n_test
                            
                            # Get Gaussian count
                            num_gaussians = len(sugar.points)
                            
                            # Create enhanced test results table (6 metrics including LPIPS)
                            test_table = Table(show_header=True, header_style="bold magenta", border_style="cyan")
                            test_table.add_column("L1 Loss", style="yellow", justify="right")
                            test_table.add_column("PSNR (dB)", style="green", justify="right")
                            test_table.add_column("SSIM", style="cyan", justify="right")
                            test_table.add_column("LPIPS", style="magenta", justify="right")
                            test_table.add_column("Gauss (K)", style="blue", justify="right")
                            test_table.add_column("Render ms", style="dim", justify="right")
                            test_table.add_row(
                                f"{test_loss:.6f}",
                                f"{test_psnr:.2f}",
                                f"{test_ssim:.3f}",
                                f"{test_lpips:.3f}",
                                f"{num_gaussians/1000.0:.1f}K",
                                f"{avg_render_time:.1f}"
                            )
                            
                            CONSOLE.print(test_table)
                            
                            # Log to TensorBoard
                            tb_writer.add_scalar('Test/loss', test_loss, iteration)
                            tb_writer.add_scalar('Test/PSNR_BUGGY_IGNORE', test_psnr, iteration)  # Marked as buggy
                            tb_writer.add_scalar('Test/ssim', test_ssim, iteration)
                            tb_writer.add_scalar('Test/lpips', test_lpips, iteration)
                            tb_writer.add_scalar('Test/render_time_ms', avg_render_time, iteration)
                            tb_writer.add_scalar('Test/num_gaussians_K', num_gaussians/1000.0, iteration)
                    
                    sugar.train()
                    print()  # Newline after table
                
                # Save model
                if (iteration % save_model_every_n_iterations == 0) or (iteration in save_milestones):
                    model_path = os.path.join(sugar_checkpoint_path, f'{iteration}.pt')
                    sugar.save_model(path=model_path,
                                    train_losses=train_losses,
                                    epoch=epoch,
                                    iteration=iteration,
                                    optimizer_state_dict=optimizer.state_dict(),
                                    )
                    # if optimize_triangles and iteration >= optimize_triangles_from:
                    #     rm.save_model(os.path.join(rc_checkpoint_path, f'rm_{iteration}.pt'))
                    
                    # Clear CUDA cache after saving to prevent fragmentation
                    torch.cuda.empty_cache()
                    
                    # Log checkpoint with size
                    if os.path.exists(model_path):
                        checkpoint_size_mb = os.path.getsize(model_path) / (1024**2)
                        print()  # Newline before checkpoint message
                        CONSOLE.print(f"💾 [green]Iter {iteration}: Saving Checkpoint[/green]")
                        CONSOLE.print(f"  → [cyan]{model_path}[/cyan] [dim]({checkpoint_size_mb:.1f} MB)[/dim]")
                        
                        # Flush TensorBoard after checkpoint save to ensure data persistence
                        tb_writer.flush()
                
                if iteration >= num_iterations:
                    # if iteration <= start_iteration + 5:
                    #     CONSOLE.print(f"[red]DEBUG: BREAKING! iteration={iteration}, num_iterations={num_iterations}[/red]")
                    break
                
                if do_sh_warmup and (iteration > 0) and (current_sh_levels < sh_levels) and (iteration % sh_warmup_every == 0):
                    current_sh_levels += 1
                    CONSOLE.print("Increasing number of spherical harmonics levels to", current_sh_levels)
                
                if do_resolution_warmup and (iteration > 0) and (current_resolution_factor > 1) and (iteration % resolution_warmup_every == 0):
                    current_resolution_factor /= 2.
                    nerfmodel.downscale_output_resolution(1/2)
                    CONSOLE.print(f'\nCamera resolution scaled to '
                            f'{nerfmodel.training_cameras.ns_cameras.height[0].item()} x '
                            f'{nerfmodel.training_cameras.ns_cameras.width[0].item()}'
                            )
                    sugar.adapt_to_cameras(nerfmodel.training_cameras)
                    # TODO: resize GT images
            
            epoch += 1

    except KeyboardInterrupt:
        progress.stop()
        CONSOLE.print(f"\n\n[bold yellow]⚠️  Training interrupted by user (CTRL-C)[/bold yellow]")
        CONSOLE.print(f"[cyan]📊 Progress:[/cyan] Completed {iteration}/{num_iterations} iterations ({100*iteration/num_iterations:.1f}%)")
        CONSOLE.print(f"[cyan]💾 Last save:[/cyan] Most recent checkpoint before interruption")
        CONSOLE.print(f"\n[bold green]✓ Exiting without saving...[/bold green]")
        
        # Close TensorBoard
        try:
            tb_writer.close()
        except:
            pass
        
        CONSOLE.print(f"\n[bold]Training interrupted. Exiting gracefully...[/bold]\n")
        return None  # Return None to indicate interrupted training

    # Stop progress bar when training completes
    progress.stop()
    
    # Training complete message
    if loss is not None:
        CONSOLE.print(f"Training finished after {num_iterations} iterations with loss={loss.detach().item()}.")
    else:
        CONSOLE.print(f"Training already complete at {num_iterations} iterations (resumed from checkpoint).")
    
    # Close TensorBoard writer
    tb_writer.close()
    CONSOLE.print(f"[bold cyan]TensorBoard logs saved to:[/bold cyan] {tb_log_dir}")
    
    CONSOLE.print("Saving final model...")
    model_path = os.path.join(sugar_checkpoint_path, f'{iteration}.pt')
    sugar.save_model(path=model_path,
                    train_losses=train_losses,
                    epoch=epoch,
                    iteration=iteration,
                    optimizer_state_dict=optimizer.state_dict(),
                    )

    CONSOLE.print("Final model saved.")
    return model_path