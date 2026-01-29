#!/usr/bin/env python3
"""
Bypass CUDA version check for simple-knn compilation.
Needed because CUDA 13.1 is installed but PyTorch built with 12.8.
"""
import sys
import torch.utils.cpp_extension

# Monkey-patch the CUDA version check
torch.utils.cpp_extension._check_cuda_version = lambda *args, **kwargs: None

# Now run setup.py
import subprocess
result = subprocess.run([sys.executable, "setup.py", "install"], env={
    **__import__('os').environ,
    'TORCH_CUDA_ARCH_LIST': '12.0',
    'CXXFLAGS': '-O2'
})
sys.exit(result.returncode)
