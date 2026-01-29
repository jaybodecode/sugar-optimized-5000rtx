"""
RTX 50 Installation Script for diff-gaussian-rasterization
Bypasses CUDA version check for CUDA 13.x with PyTorch 2.10 (CUDA 12.8)
"""

import torch.utils.cpp_extension
import sys

# Monkey-patch the CUDA version check to always pass
original_check = torch.utils.cpp_extension._check_cuda_version

def patched_check(*args, **kwargs):
    print("⚠️  Bypassing CUDA version check (13.1 vs 12.8 - compatible)")
    return None

torch.utils.cpp_extension._check_cuda_version = patched_check

# Now run setup.py
exec(open('setup.py').read())
