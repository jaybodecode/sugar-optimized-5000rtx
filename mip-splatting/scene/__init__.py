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
import random
import json
import sys

# Add parent directory to path for console_logger
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import console_logger as log

from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            log.log(f"Loading trained model at iteration {self.loaded_iter}")

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.test_camera_count)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            log.log("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "metadata.json")):
            log.log("Found metadata.json file, assuming multi scale Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Multi-scale"](args.source_path, args.white_background, args.eval, args.load_allres)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            # random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        
        # Initialize camera cache settings ONCE before loading any cameras
        # NOTE: Don't clear cache here - it may contain references from previous Scene instances
        from scene.cameras import Camera
        
        if getattr(args, 'low_dram', False):
            # Calculate cache size based on image resolution
            cache_gb = getattr(args, 'image_cache_gb', 2.0)
            if scene_info.train_cameras:
                first_cam = scene_info.train_cameras[0]
                if hasattr(first_cam.image, 'size'):
                    orig_w, orig_h = first_cam.image.size
                else:
                    orig_w, orig_h = first_cam.width, first_cam.height
                
                resolution = getattr(args, 'resolution', -1)
                if resolution == -1:
                    scaled_w, scaled_h = orig_w, orig_h
                else:
                    scaled_w = orig_w // resolution
                    scaled_h = orig_h // resolution
                
                bytes_per_image = scaled_w * scaled_h * 3 * 4
                gb_per_image = bytes_per_image / (1024**3)
                cache_size = max(1, int(cache_gb / gb_per_image))
                
                log.log(f"[Lazy Loading] Resolution: {scaled_w}×{scaled_h}, ~{bytes_per_image/(1024**2):.1f}MB per image (uncompressed tensor in RAM)")
                log.log(f"[Lazy Loading] Cache limit: {cache_gb:.1f}GB → {cache_size} images")
            else:
                cache_size = 20
            Camera.set_cache_size(cache_size)
        else:
            # Eager loading - set large cache size to prevent issues
            Camera.set_cache_size(10000)

        for resolution_scale in resolution_scales:
            log.log("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            log.log("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]