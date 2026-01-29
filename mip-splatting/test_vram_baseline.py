#!/usr/bin/env python3
"""Simple VRAM measurement script for comparing optimizations.

Usage:
    python test_vram_baseline.py --checkpoint path/to/checkpoint --label "Baseline"
"""

import torch
import argparse
import sys
from pathlib import Path
from scene import Scene
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, PipelineParams
from utils.general_utils import safe_state
import gc

def get_vram_mb():
    """Get current VRAM usage in MB."""
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / (1024**2)

def measure_vram(checkpoint_path, dataset_path, label="Test"):
    """Measure VRAM usage during evaluation."""
    
    print(f"\n{'='*60}")
    print(f"VRAM Test: {label}")
    print(f"{'='*60}")
    
    # Clear everything first
    torch.cuda.empty_cache()
    gc.collect()
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    args, _ = parser.parse_known_args([
        '-s', str(dataset_path),
        '-m', str(Path(checkpoint_path).parent),
        '--eval'
    ])
    
    # Initialize model
    gaussians = GaussianModel(3)
    
    print(f"✓ Loading checkpoint: {checkpoint_path}")
    (model_params, first_iter) = torch.load(checkpoint_path)
    gaussians.restore(model_params, None)
    
    vram_after_load = get_vram_mb()
    print(f"  VRAM after load: {vram_after_load:.1f} MB")
    
    # Create scene
    scene_args = argparse.Namespace(**vars(args))
    scene_args.images = "images"
    scene_args.resolution = 2  # Match training resolution
    scene_args.eval = True
    
    scene = Scene(scene_args, gaussians, load_iteration=first_iter, shuffle=False)
    
    vram_after_scene = get_vram_mb()
    print(f"  VRAM after scene: {vram_after_scene:.1f} MB")
    
    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Get test cameras
    test_cams = scene.getTestCameras()
    
    print(f"  Testing with {len(test_cams)} cameras")
    
    # Peak VRAM tracker
    peak_vram = 0
    
    # Render first 5 test views to simulate eval
    for idx, viewpoint in enumerate(test_cams[:5]):
        render_pkg = render(viewpoint, gaussians, pipeline_params.extract(args), background)
        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
        
        # Simulate GT image caching (what happens during eval)
        if not hasattr(viewpoint, '_cached_gt_cuda'):
            viewpoint._cached_gt_cuda = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
        gt_image = viewpoint._cached_gt_cuda
        
        current_vram = get_vram_mb()
        peak_vram = max(peak_vram, current_vram)
        
        if idx == 0:
            print(f"  VRAM after 1st render: {current_vram:.1f} MB")
    
    final_vram = get_vram_mb()
    
    print(f"\n{'─'*60}")
    print(f"  Peak VRAM:  {peak_vram:.1f} MB")
    print(f"  Final VRAM: {final_vram:.1f} MB")
    print(f"{'─'*60}\n")
    
    # Cleanup
    del scene, gaussians, render_pkg, image, gt_image, background
    torch.cuda.empty_cache()
    
    return {
        'label': label,
        'after_load': vram_after_load,
        'after_scene': vram_after_scene,
        'peak': peak_vram,
        'final': final_vram
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure VRAM usage with checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint.pth")
    parser.add_argument("--dataset", type=str, default="../SAMPLES/garden", help="Dataset path")
    parser.add_argument("--label", type=str, default="Test", help="Label for this test")
    
    args = parser.parse_args()
    
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    result = measure_vram(args.checkpoint, args.dataset, args.label)
    
    print(f"✓ Test complete: {result['label']}")
    print(f"  Peak VRAM reduction target: 500MB (register spilling) → 3.5GB (full hybrid)")
