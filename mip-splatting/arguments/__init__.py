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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cpu"  # Default to CPU for image storage (less VRAM pressure)
        self.eval = False
        self.eval_only = False  # Skip optimizer setup for evaluation-only (saves 2-3GB VRAM when loading checkpoint)
        self._kernel_size = 0.1
        # self.use_spatial_gaussian_bias = False
        self.ray_jitter = False
        self.resample_gt_image = False
        self.load_allres = False
        self.sample_more_highres = False
        self.low_dram = False  # Lazy loading (LRU cache) - default is eager loading (all images in RAM)
        self.image_cache_gb = 2.0  # LRU cache memory limit in GB (auto-calculates image count based on resolution)
        self.test_camera_count = 6  # Number of cameras for test set (evenly distributed, includes first/last frame)
        self.delete_first = False  # Auto-delete existing output folder without prompting (useful for scripts/automation)
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 500
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 8_000  # Emergency brake: Stop at 8k (not 12k) to limit Gaussian growth on 16GB VRAM
        self.densify_grad_threshold = 0.0002
        self.min_opacity_threshold = 0.03  # Emergency brake: 3x more aggressive (sigmoid ≈ 51.25% vs 50.25% at 0.01)
        
        # Binary search hard cap (prevents VRAM overflow by enforcing absolute Gaussian count limit)
        self.target_gaussian_count = 5_000_000  # Enforce 5M cap for RTX 5060 Ti 16GB (set 0 to disable, 8M for 24GB)
        self.binary_search_min = 0.003  # Lower bound - start higher for faster convergence
        self.binary_search_max = 0.08  # Upper bound - lower than 0.1 to be more conservative
        self.binary_search_iterations = 12  # Precision control: 12 iterations = good balance (15 = max precision)
        
        self.enable_lpips = True  # Perceptual loss metric (adds ~1min per evaluation, disable with --enable_lpips False)
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
