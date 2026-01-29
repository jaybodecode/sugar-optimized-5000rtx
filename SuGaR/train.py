import argparse
import warnings
# Suppress deprecation warnings from PyTorch/torchvision (don't affect performance)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
from sugar_utils.general_utils import str2bool
from sugar_trainers.coarse_sdf import coarse_training_with_sdf_regularization
from sugar_trainers.coarse_density_and_dn_consistency import coarse_training_with_density_regularization_and_dn_consistency
from sugar_extractors.coarse_mesh import extract_mesh_from_coarse_sugar
from sugar_trainers.refine import refined_training
from sugar_extractors.refined_mesh import extract_mesh_and_texture_from_refined_sugar


class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self


if __name__ == "__main__":
    # ----- Parser -----
    parser = argparse.ArgumentParser(description='Script to optimize a full SuGaR model.')
    
    # Data and vanilla 3DGS checkpoint
    parser.add_argument('-s', '--scene_path',
                        type=str, 
                        help='(Required) path to the scene data to use.')  
    parser.add_argument('-c', '--checkpoint_path',
                        type=str, 
                        help='(Required) path to the vanilla 3D Gaussian Splatting Checkpoint to load.')
    parser.add_argument('-i', '--iteration_to_load', 
                        type=int, default=7000, 
                        help='iteration to load.')
    
    # Regularization for coarse SuGaR
    parser.add_argument('-r', '--regularization_type', type=str,
                        help='(Required) Type of regularization to use for coarse SuGaR. Can be "sdf", "density" or "dn_consistency". '
                        'Recommended: "density" for 16GB VRAM (excellent quality, 73%% VRAM), '
                        '"dn_consistency" for 24GB+ VRAM (best quality with depth-normal supervision).')
    
    # Extract mesh
    parser.add_argument('-l', '--surface_level', type=float, default=0.3, 
                        help='Surface level to extract the mesh at. Default is 0.3')
    parser.add_argument('-v', '--n_vertices_in_mesh', type=int, default=1_000_000, 
                        help='Number of vertices in the extracted mesh.')
    parser.add_argument('--project_mesh_on_surface_points', type=str2bool, default=True, 
                        help='If True, project the mesh on the surface points for better details.')
    parser.add_argument('-b', '--bboxmin', type=str, default=None, 
                        help='Min coordinates to use for foreground.')  
    parser.add_argument('-B', '--bboxmax', type=str, default=None, 
                        help='Max coordinates to use for foreground.')
    parser.add_argument('--center_bbox', type=str2bool, default=True, 
                        help='If True, center the bbox. Default is False.')
    
    # Parameters for refined SuGaR
    parser.add_argument('-g', '--gaussians_per_triangle', type=int, default=1, 
                        help='Number of gaussians per triangle.')
    parser.add_argument('-f', '--refinement_iterations', type=int, default=15_000, 
                        help='Number of refinement iterations.')
    
    # (Optional) Parameters for textured mesh extraction
    parser.add_argument('-t', '--export_uv_textured_mesh', type=str2bool, default=True, 
                        help='If True, will export a textured mesh as an .obj file from the refined SuGaR model. '
                        'Computing a traditional colored UV texture should take less than 10 minutes.')
    parser.add_argument('--square_size',
                        default=8, type=int, help='Size of the square to use for the UV texture.')
    parser.add_argument('--postprocess_mesh', type=str2bool, default=False, 
                        help='If True, postprocess the mesh by removing border triangles with low-density. '
                        'This step takes a few minutes and is not needed in general, as it can also be risky. '
                        'However, it increases the quality of the mesh in some cases, especially when an object is visible only from one side.')
    parser.add_argument('--postprocess_density_threshold', type=float, default=0.1,
                        help='Threshold to use for postprocessing the mesh.')
    parser.add_argument('--postprocess_iterations', type=int, default=5,
                        help='Number of iterations to use for postprocessing the mesh.')
    
    # (Optional) PLY file export
    parser.add_argument('--export_ply', type=str2bool, default=True,
                        help='If True, export a ply file with the refined 3D Gaussians at the end of the training. '
                        'This file can be large (+/- 500MB), but is needed for using the dedicated viewer. Default is True.')
    
    # (Optional) Skip training if checkpoint exists
    parser.add_argument('--skip_training', type=str2bool, default=False,
                        help='If True, skip coarse training and go straight to mesh extraction if checkpoint exists. '
                        'Useful for testing mesh extraction/refinement without waiting for slow training.')
    
    # (Optional) Default configurations
    parser.add_argument('--low_poly', type=str2bool, default=False, 
                        help='Use standard config for a low poly mesh, with 200k vertices and 6 Gaussians per triangle.')
    parser.add_argument('--high_poly', type=str2bool, default=False,
                        help='Use standard config for a high poly mesh, with 1M vertices and 1 Gaussians per triangle.')
    parser.add_argument('--refinement_time', type=str, default=None, 
                        help="Default configs for time to spend on refinement. Can be 'short', 'medium' or 'long'.")
      
    # Evaluation split
    parser.add_argument('--eval', type=str2bool, default=True, help='Use eval split.')

    # GPU
    parser.add_argument('--gpu', type=int, default=0, help='Index of GPU device to use.')
    parser.add_argument('--white_background', type=str2bool, default=False, help='Use a white background instead of black.')
    
    # VRAM Optimization
    parser.add_argument('--full_res_normals', type=str2bool, default=True,
                        help='Use full resolution for depth-normal consistency. '
                        'Default True (enables normal checkpointing for max VRAM savings). '
                        'Set False for half-res (slightly faster, ~1-3%% quality trade-off, cannot checkpoint normals).')
    parser.add_argument('--use_gradient_checkpointing', type=str2bool, default=True,
                        help='Use gradient checkpointing (Phase 2 optimization). '
                        'Default True (recompute activations instead of storing, saves 30-40%% VRAM). '
                        'Set to False for 24GB+ VRAM (no checkpointing, 30-50%% faster training).')
    parser.add_argument('--mesh_extraction_resolution_factor', type=float, default=0.5,
                        help='Resolution factor for mesh extraction (post-training stage). '
                        'Default 0.5 (half-res, ~15GB VRAM for 16GB GPUs). '
                        'Set 1.0 for full resolution (24GB VRAM, slightly better mesh quality on 32GB+ GPUs).')
    
    # Training Monitoring
    parser.add_argument('--checkpoint_interval', type=int, default=1000,
                        help='Save checkpoint every N iterations. Default 1000. Set to 0 to disable interval checkpoints.')
    parser.add_argument('--checkpoint_milestones', type=int, nargs='+', default=[7000, 9000, 12000, 15000],
                        help='Specific iterations at which to save checkpoints (in addition to interval). '
                        'Example: --checkpoint_milestones 7000 10000 15000')
    parser.add_argument('--test_iterations', type=int, nargs='+', default=[7000, 10000, 15000],
                        help='Iterations at which to run test evaluation and log to TensorBoard. '
                        'Example: --test_iterations 7000 10000 15000')
    parser.add_argument('--coarse_iterations', type=int, default=15000,
                        help='Total iterations for coarse training. Default: 15000 (SuGaR standard). '
                        'Increase to 20000-25000 to find point of diminishing returns.')
    
    # Resume from checkpoint
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to SuGaR checkpoint (.pt file) to resume training from. '
                        'Example: path/to/checkpoint/15000.pt')
    
    # Experiment naming
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for this training run. Creates output in <checkpoint_path>_mesh/<experiment_name>/')
    parser.add_argument('--delete_first', action='store_true',
                        help='Automatically delete existing output folder without prompting (useful for conda run)')
    
    # Tensor memory profiling
    parser.add_argument('--profile_tensors', type=str2bool, default=False,
                        help='Enable detailed tensor memory profiling to TensorBoard. '
                        'Logs tensor breakdown every 50 iterations (adds ~50-200ms overhead per log). '
                        'Creates Memory/Tensor_* graphs showing categories, dtypes, and large tensors. '
                        'Default False (disabled).')

    # Parse arguments
    args = parser.parse_args()
    
    # Handle output path with _mesh suffix and experiment_name
    import sys
    import shutil
    from datetime import datetime
    import os
    
    if args.checkpoint_path:
        # Remove trailing slash for consistency
        base_checkpoint = args.checkpoint_path.rstrip('/')
        
        # Add _mesh suffix to checkpoint path
        mesh_output_base = f"{base_checkpoint}_mesh"
        
        # Use experiment_name if provided, otherwise use timestamp
        if args.experiment_name:
            final_output_path = os.path.join(mesh_output_base, args.experiment_name)
        else:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_output_path = os.path.join(mesh_output_base, timestamp_str)
        
        # Check if output folder already exists
        if os.path.exists(final_output_path):
            print(f"\n⚠️  Output folder already exists: {final_output_path}")
            
            dir_contents = os.listdir(final_output_path)
            if dir_contents:
                print(f"   Folder contains {len(dir_contents)} items")
            
            if args.skip_training:
                print("--skip_training enabled: Keeping existing folder (no prompt)...")
            elif args.delete_first:
                print("--delete_first enabled: Auto-deleting existing folder...")
                shutil.rmtree(final_output_path)
                print("✓ Folder deleted")
            else:
                response = input("Delete existing folder and continue? [y/N]: ").strip().lower()
                
                if response == 'y' or response == 'yes':
                    print("Deleting existing folder...")
                    shutil.rmtree(final_output_path)
                    print("✓ Folder deleted")
                else:
                    print("⚠️  Keeping existing folder - contents may be overwritten during training")
                    print("Tip: Use --experiment_name with a unique name, or --delete_first for auto-delete")
        
        # Create the output directory
        os.makedirs(final_output_path, exist_ok=True)
        print(f"\n📁 Output folder: {final_output_path}\n")
        
        # Override output_dir for all stages to use this path
        args.output_dir_override = final_output_path
    else:
        args.output_dir_override = None
    
    if args.low_poly:
        args.n_vertices_in_mesh = 200_000
        args.gaussians_per_triangle = 6
        print('Using low poly config.')
    if args.high_poly:
        args.n_vertices_in_mesh = 1_000_000
        args.gaussians_per_triangle = 1
        print('Using high poly config.')
    if args.refinement_time == 'short':
        args.refinement_iterations = 2_000
        print('Using short refinement time.')
    if args.refinement_time == 'medium':
        args.refinement_iterations = 7_000
        print('Using medium refinement time.')
    if args.refinement_time == 'long':
        args.refinement_iterations = 15_000
        print('Using long refinement time.')
    if args.export_uv_textured_mesh:
        print('Will export a UV-textured mesh as an .obj file.')
    if args.export_ply:
        print('Will export a ply file with the refined 3D Gaussians at the end of the training.')
    
    # ----- Optimize coarse SuGaR (or skip if checkpoint exists) -----
    # Map -r flag to use_dn_consistency (backward compatible)
    use_dn_consistency = (args.regularization_type == 'dn_consistency')
    
    coarse_args = AttrDict({
        'checkpoint_path': args.checkpoint_path,
        'scene_path': args.scene_path,
        'iteration_to_load': args.iteration_to_load,
        'output_dir': args.output_dir_override if hasattr(args, 'output_dir_override') else None,
        'eval': args.eval,
        'estimation_factor': 0.2,
        'normal_factor': 0.2,
        'gpu': args.gpu,
        'white_background': args.white_background,
        'full_res_normals': args.full_res_normals,
        'use_gradient_checkpointing': args.use_gradient_checkpointing,
        'checkpoint_interval': args.checkpoint_interval,
        'checkpoint_milestones': args.checkpoint_milestones,
        'test_iterations': args.test_iterations,
        'coarse_iterations': args.coarse_iterations,
        'resume_checkpoint': args.resume_checkpoint,
        'use_dn_consistency': use_dn_consistency,  # Unified trainer flag
        'profile_tensors': args.profile_tensors,  # Tensor memory profiling
    })
    
    # Check if we should skip training and use existing checkpoint
    if args.skip_training and args.resume_checkpoint:
        import os
        if os.path.exists(args.resume_checkpoint):
            print(f"\n[SKIP TRAINING] Using existing checkpoint: {args.resume_checkpoint}")
            print(f"[SKIP TRAINING] Skipping to mesh extraction...\n")
            coarse_sugar_path = args.resume_checkpoint
        else:
            print(f"\n[WARNING] --skip_training=True but checkpoint not found: {args.resume_checkpoint}")
            print(f"[WARNING] Running training instead...\n")
            args.skip_training = False
    
    if not args.skip_training:
        if args.regularization_type == 'sdf':
            coarse_sugar_path = coarse_training_with_sdf_regularization(coarse_args)
        elif args.regularization_type in ['density', 'dn_consistency']:
            # Both use unified trainer (dn_consistency controlled by use_dn_consistency flag)
            coarse_sugar_path = coarse_training_with_density_regularization_and_dn_consistency(coarse_args)
        else:
            raise ValueError(f'Unknown regularization type: {args.regularization_type}')
    
    
    # ----- Extract mesh from coarse SuGaR -----
    coarse_mesh_args = AttrDict({
        'scene_path': args.scene_path,
        'checkpoint_path': args.checkpoint_path,
        'iteration_to_load': args.iteration_to_load,
        'coarse_model_path': coarse_sugar_path,
        'surface_level': args.surface_level,
        'decimation_target': args.n_vertices_in_mesh,
        'project_mesh_on_surface_points': args.project_mesh_on_surface_points,
        'mesh_output_dir': args.output_dir_override if hasattr(args, 'output_dir_override') else None,
        'bboxmin': args.bboxmin,
        'bboxmax': args.bboxmax,
        'center_bbox': args.center_bbox,
        'gpu': args.gpu,
        'eval': args.eval,
        'use_centers_to_extract_mesh': False,
        'use_marching_cubes': False,
        'use_vanilla_3dgs': False,
        'mesh_extraction_resolution_factor': args.mesh_extraction_resolution_factor,
    })
    coarse_mesh_path = extract_mesh_from_coarse_sugar(coarse_mesh_args)[0]
    
    
    # ----- Refine SuGaR -----
    refined_args = AttrDict({
        'scene_path': args.scene_path,
        'checkpoint_path': args.checkpoint_path,
        'mesh_path': coarse_mesh_path,      
        'output_dir': args.output_dir_override if hasattr(args, 'output_dir_override') else None,
        'iteration_to_load': args.iteration_to_load,
        'normal_consistency_factor': 0.1,    
        'gaussians_per_triangle': args.gaussians_per_triangle,        
        'n_vertices_in_fg': args.n_vertices_in_mesh,
        'refinement_iterations': args.refinement_iterations,
        'bboxmin': args.bboxmin,
        'bboxmax': args.bboxmax,
        'export_ply': args.export_ply,
        'eval': args.eval,
        'gpu': args.gpu,
        'white_background': args.white_background,
    })
    refined_sugar_path = refined_training(refined_args)
    
    
    # ----- Extract mesh and texture from refined SuGaR -----
    if args.export_uv_textured_mesh:
        refined_mesh_args = AttrDict({
            'scene_path': args.scene_path,
            'iteration_to_load': args.iteration_to_load,
            'checkpoint_path': args.checkpoint_path,
            'refined_model_path': refined_sugar_path,
            'mesh_output_dir': args.output_dir_override if hasattr(args, 'output_dir_override') else None,
            'n_gaussians_per_surface_triangle': args.gaussians_per_triangle,
            'square_size': args.square_size,
            'eval': args.eval,
            'gpu': args.gpu,
            'postprocess_mesh': args.postprocess_mesh,
            'postprocess_density_threshold': args.postprocess_density_threshold,
            'postprocess_iterations': args.postprocess_iterations,
        })
        refined_mesh_path = extract_mesh_and_texture_from_refined_sugar(refined_mesh_args)
        