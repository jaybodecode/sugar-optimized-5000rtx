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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from functools import lru_cache
from collections import OrderedDict

class Camera(nn.Module):
    # Class-level LRU cache for lazy-loaded images
    # Keeps only the last N images in RAM to prevent memory buildup
    _image_cache = OrderedDict()  # {image_name: tensor}
    _cache_size = 20  # Keep last 20 images in RAM (tune based on available RAM and batch size)
    _cache_enabled = False  # Only enable cache when lazy loading is actually used
    
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 image_path=None, resolution=None, lazy_load=False
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # Lazy loading support for low RAM systems
        self.lazy_load = lazy_load
        self.image_path = image_path
        self.resolution = resolution
        self._cached_image = None
        
        # Set dimensions first (before setting original_image)
        self.image_width = image.shape[2]
        self.image_height = image.shape[1]
        
        # Calculate focal lengths (needed by original_image setter)
        self.tan_fovx = np.tan(self.FoVx / 2.0)
        self.tan_fovy = np.tan(self.FoVy / 2.0)
        self.focal_y = self.image_height / (2.0 * self.tan_fovy)
        self.focal_x = self.image_width / (2.0 * self.tan_fovx)
        
        if lazy_load:
            # Don't load image into RAM yet - will load on-demand
            pass
        else:
            # Original eager loading behavior
            self.original_image = image.clamp(0.0, 1.0).to(self.data_device)

            if gt_alpha_mask is not None:
                self.original_image *= gt_alpha_mask.to(self.data_device)
            else:
                self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    
    def get_image(self):
        """Get image - loads on-demand if lazy_load=True with LRU caching.
        
        Uses class-level LRU cache to limit RAM usage. Only keeps last N images
        in memory, evicting oldest when cache is full. Prevents RAM buildup with
        high-resolution images.
        """
        if self.lazy_load and Camera._cache_enabled:
            cache_key = self.image_name
            
            # Check if image is in class-level cache
            if cache_key in Camera._image_cache:
                # Move to end (mark as recently used)
                Camera._image_cache.move_to_end(cache_key)
                return Camera._image_cache[cache_key]
            
            # Load image from disk
            from PIL import Image
            from utils.general_utils import PILtoTorch
            
            pil_image = Image.open(self.image_path)
            resized_image_rgb = PILtoTorch(pil_image, self.resolution)
            gt_image = resized_image_rgb[:3, ...]
            loaded_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
            
            # Handle alpha mask if present
            if resized_image_rgb.shape[0] == 4:
                loaded_mask = resized_image_rgb[3:4, ...]
                loaded_image *= loaded_mask.to(self.data_device)
            else:
                loaded_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            
            # Add to class-level cache
            Camera._image_cache[cache_key] = loaded_image
            
            # Evict oldest if cache exceeds size limit
            while len(Camera._image_cache) > Camera._cache_size:
                oldest_key, _ = Camera._image_cache.popitem(last=False)
            
            return loaded_image
        else:
            return self.original_image
    
    # Property for backward compatibility
    @property
    def original_image(self):
        if self.lazy_load:
            return self.get_image()
        return self._original_image
    
    @original_image.setter
    def original_image(self, value):
        self._original_image = value
    
    @staticmethod
    def clear_image_cache():
        """Clear the image cache to free RAM/VRAM."""
        Camera._image_cache.clear()
        Camera._cache_enabled = False
    
    @staticmethod
    def set_cache_size(size):
        """Set the LRU cache size (number of images to keep in RAM).
        
        Args:
            size: Number of images to cache. Automatically calculated from
                  --image_cache_gb parameter based on actual image resolution.
        """
        Camera._cache_size = size
        Camera._cache_enabled = (size > 0 and size < 10000)  # Only enable cache for lazy loading
         
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

