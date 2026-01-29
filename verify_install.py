#!/usr/bin/env python3
"""
Installation Verification Script - SuGaR Optimized for RTX 5000 Series
Combines system checks, GPU validation, and dependency verification
"""

import sys

def test_python_version():
    """Check Python version"""
    print("\n" + "="*60)
    print("🐍 Python Version Check")
    print("="*60)
    
    version = sys.version_info
    print(f"  Python: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 10:
        print(f"  ✅ Python {version.major}.{version.minor} is supported")
        return True
    else:
        print(f"  ❌ Python {version.major}.{version.minor} not supported (need 3.10+)")
        return False

def test_cuda_availability():
    """Test basic CUDA availability"""
    print("\n" + "="*60)
    print("🔍 CUDA Availability Test")
    print("="*60)
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        
        if not torch.cuda.is_available():
            print("❌ CUDA not available!")
            return False
            
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ cuDNN version: {torch.backends.cudnn.version()}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_gpu_properties():
    """Test GPU detection and properties"""
    print("\n" + "="*60)
    print("🎮 GPU Properties")
    print("="*60)
    
    try:
        import torch
        
        gpu_count = torch.cuda.device_count()
        print(f"✓ GPU count: {gpu_count}")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            compute_cap = torch.cuda.get_device_capability(i)
            memory_gb = props.total_memory / (1024**3)
            
            print(f"\n  GPU {i}:")
            print(f"    Name: {props.name}")
            print(f"    Compute Capability: sm_{compute_cap[0]}{compute_cap[1]}")
            print(f"    VRAM: {memory_gb:.2f} GB")
            print(f"    SMs: {props.multi_processor_count}")
            print(f"    Max threads/block: {props.max_threads_per_block}")
            
            # Check for RTX 5000 series (compute 12.0)
            if compute_cap == (12, 0):
                print(f"    ✅ RTX 5000 Series (Blackwell) detected!")
            elif compute_cap[0] >= 9:
                print(f"    ✅ RTX 50 Series detected!")
            elif compute_cap[0] >= 8:
                print(f"    ⚠️  RTX 30/40 Series (compute {compute_cap[0]}.{compute_cap[1]})")
                print(f"    ⚠️  This project is optimized for RTX 5000 series")
            else:
                print(f"    ⚠️  Older GPU architecture (compute {compute_cap[0]}.{compute_cap[1]})")
                print(f"    ⚠️  May not support all optimizations")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_pytorch_version():
    """Test PyTorch version for sm_120 support"""
    print("\n" + "="*60)
    print("📦 PyTorch Version Check (Blackwell Support)")
    print("="*60)
    
    try:
        import torch
        
        version = torch.__version__
        cuda_version = torch.version.cuda
        
        print(f"  PyTorch version: {version}")
        print(f"  CUDA version: {cuda_version}")
        
        # Check compute capability
        if torch.cuda.is_available():
            compute_cap = torch.cuda.get_device_capability(0)
            print(f"  GPU compute capability: sm_{compute_cap[0]}{compute_cap[1]}")
            
            # RTX 5000 series is sm_120 (Blackwell)
            if compute_cap == (12, 0):
                print(f"\n  ✅ RTX 5000 Series (Blackwell sm_120) detected")
                
                # Check if PyTorch version supports it
                version_parts = version.split('+')[0].split('.')
                major = int(version_parts[0])
                minor = int(version_parts[1])
                
                if major >= 2 and minor >= 11 and cuda_version >= "13.0":
                    print(f"  ✅ PyTorch {version} with CUDA {cuda_version} supports sm_120!")
                    return True
                else:
                    print(f"  ⚠️  PyTorch {version} with CUDA {cuda_version} may lack full sm_120 support")
                    print(f"  📝 Recommended: PyTorch 2.11+ with CUDA 13.0")
                    return False
            else:
                print(f"  ⚠️  Not an RTX 5000 series GPU")
                print(f"  ⚠️  This project is optimized for Blackwell architecture")
                return True  # Don't fail, just warn
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_core_dependencies():
    """Test core Python dependencies"""
    print("\n" + "="*60)
    print("📚 Core Dependencies Check")
    print("="*60)
    
    modules = {
        'plyfile': 'PLY file I/O',
        'tqdm': 'Progress bars',
        'tensorboard': 'Training visualization',
        'cv2': 'OpenCV (opencv-python)',
        'PIL': 'Pillow',
        'skimage': 'scikit-image',
        'trimesh': 'Mesh processing',
        'open3d': 'Point cloud processing',
        'lpips': 'Perceptual loss',
        'rich': 'Terminal formatting',
    }
    
    all_ok = True
    for module, description in modules.items():
        try:
            __import__(module)
            print(f"  ✓ {module:15s} - {description}")
        except ImportError:
            print(f"  ❌ {module:15s} - {description} (MISSING)")
            all_ok = False
    
    return all_ok

def test_pytorch3d():
    """Test pytorch3d installation"""
    print("\n" + "="*60)
    print("🔺 pytorch3d Test")
    print("="*60)
    
    try:
        import pytorch3d
        version = getattr(pytorch3d, '__version__', 'editable install')
        print(f"  ✓ pytorch3d version: {version}")
        
        # Test basic operations
        from pytorch3d.structures import Meshes
        from pytorch3d.ops import knn_points
        import torch
        
        print(f"  ✓ Meshes class available")
        print(f"  ✓ KNN operations available")
        
        # Test CUDA KNN (critical for SuGaR)
        points = torch.rand(100, 3).cuda()
        knn_result = knn_points(points.unsqueeze(0), points.unsqueeze(0), K=3)
        print(f"  ✓ CUDA KNN working: {knn_result.dists.shape}")
        
        return True
    except ImportError:
        print(f"  ❌ pytorch3d not installed")
        print(f"  📝 Install: cd pytorch3d && pip install --no-build-isolation -e .")
        return False
    except Exception as e:
        print(f"  ❌ pytorch3d error: {e}")
        return False

def test_nvdiffrast():
    """Test nvdiffrast installation"""
    print("\n" + "="*60)
    print("🎨 nvdiffrast Test")
    print("="*60)
    
    try:
        import nvdiffrast.torch as dr
        print(f"  ✓ nvdiffrast imported successfully")
        
        # Test rasterizer context
        import torch
        glctx = dr.RasterizeGLContext()
        print(f"  ✓ OpenGL context created")
        
        return True
    except ImportError:
        print(f"  ❌ nvdiffrast not installed")
        print(f"  📝 Install: cd nvdiffrast && pip install --no-build-isolation .")
        return False
    except Exception as e:
        print(f"  ❌ nvdiffrast error: {e}")
        return False

def test_simple_knn():
    """Test simple-knn CUDA module (critical for Gaussian Splatting)"""
    print("\n" + "="*60)
    print("🔍 simple-knn Test")
    print("="*60)
    
    try:
        import torch
        from simple_knn._C import distCUDA2
        
        print("  ✓ simple-knn module imported")
        
        # Test with small dataset
        points = torch.rand(1000, 3).cuda()
        distances = distCUDA2(points)
        
        print(f"  ✓ distCUDA2 executed successfully!")
        print(f"  ✓ Output shape: {distances.shape}")
        
        return True
        
    except ImportError:
        print("  ❌ simple-knn not installed")
        print("  📝 Install: cd mip-splatting/submodules/simple-knn && pip install -e .")
        return False
    except Exception as e:
        print(f"  ❌ simple-knn test failed: {e}")
        print("\n  ⚠️  This usually means PyTorch doesn't support sm_120")
        print("  📝 Upgrade: pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130")
        return False

def test_rtx_optimizations():
    """Test RTX 5000 specific optimizations"""
    print("\n" + "="*60)
    print("⚡ RTX Optimizations Check")
    print("="*60)
    
    try:
        import torch
        
        # Enable optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        print("✓ TF32 enabled for matmul:", torch.backends.cuda.matmul.allow_tf32)
        print("✓ TF32 enabled for cuDNN:", torch.backends.cudnn.allow_tf32)
        print("✓ cuDNN benchmark mode:", torch.backends.cudnn.benchmark)
        
        # Test BF16 support (better for RTX 5000)
        if torch.cuda.is_bf16_supported():
            print("✓ BF16 (bfloat16) supported")
        else:
            print("⚠️  BF16 not supported")
        
        # Test FP8 (RTX 50 Blackwell feature)
        if hasattr(torch, 'float8_e4m3fn'):
            print("✓ FP8 dtypes available (RTX 50 Blackwell)")
        else:
            print("ℹ️  FP8 dtypes not available (requires PyTorch 2.11+)")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_memory():
    """Test memory allocation"""
    print("\n" + "="*60)
    print("💾 Memory Test")
    print("="*60)
    
    try:
        import torch
        
        device = torch.device('cuda:0')
        
        # Small allocation test
        x = torch.randn(1000, 1000, device=device)
        print(f"✓ Allocated 1000x1000 tensor")
        print(f"  Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"  Memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        
        # Check total VRAM
        props = torch.cuda.get_device_properties(0)
        total_vram_gb = props.total_memory / (1024**3)
        
        if total_vram_gb >= 16:
            print(f"✓ {total_vram_gb:.1f} GB VRAM - sufficient for training")
        elif total_vram_gb >= 12:
            print(f"⚠️  {total_vram_gb:.1f} GB VRAM - may need smaller datasets")
        else:
            print(f"❌ {total_vram_gb:.1f} GB VRAM - insufficient (need 16GB+)")
        
        del x
        torch.cuda.empty_cache()
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_tensor_operations():
    """Test basic tensor operations and benchmarks"""
    print("\n" + "="*60)
    print("🧮 Tensor Operations Benchmark")
    print("="*60)
    
    try:
        import torch
        import time
        
        device = torch.device('cuda:0')
        
        # Matrix multiplication benchmark
        size = 4096
        print(f"\n  Testing {size}x{size} matrix multiplication...")
        
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        # Warmup
        for _ in range(3):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(10):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        tflops = (2 * size**3 * 10) / (elapsed * 1e12)
        
        print(f"  ✓ Time: {elapsed:.3f}s (10 iterations)")
        print(f"  ✓ Performance: {tflops:.2f} TFLOPS")
        print(f"  ✓ Avg per matmul: {elapsed/10*1000:.2f}ms")
        
        # Test FP16 if supported
        if torch.cuda.get_device_capability(0)[0] >= 8:
            print(f"\n  Testing mixed precision (FP16)...")
            a_fp16 = a.half()
            b_fp16 = b.half()
            
            start = time.time()
            for _ in range(10):
                c_fp16 = torch.matmul(a_fp16, b_fp16)
            torch.cuda.synchronize()
            elapsed_fp16 = time.time() - start
            
            speedup_fp16 = elapsed / elapsed_fp16
            print(f"  ✓ FP16 time: {elapsed_fp16:.3f}s")
            print(f"  ✓ FP16 speedup: {speedup_fp16:.2f}x")
            
        # Test BF16 (better for RTX 50 series)
        if torch.cuda.is_bf16_supported():
            print(f"\n  Testing BF16 (better for RTX 50)...")
            a_bf16 = a.bfloat16()
            b_bf16 = b.bfloat16()
            
            start = time.time()
            for _ in range(10):
                c_bf16 = torch.matmul(a_bf16, b_bf16)
            torch.cuda.synchronize()
            elapsed_bf16 = time.time() - start
            
            speedup_bf16 = elapsed / elapsed_bf16
            print(f"  ✓ BF16 time: {elapsed_bf16:.3f}s")
            print(f"  ✓ BF16 speedup: {speedup_bf16:.2f}x")
        
        del a, b, c
        torch.cuda.empty_cache()
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all verification tests"""
    print("\n" + "█"*60)
    print("  SuGaR Installation Verification")
    print("  Optimized for NVIDIA RTX 5000 Series")
    print("█"*60)
    
    results = []
    
    # System checks
    results.append(("Python Version", test_python_version()))
    results.append(("CUDA Availability", test_cuda_availability()))
    
    if results[-1][1]:  # Only continue if CUDA is available
        results.append(("GPU Properties", test_gpu_properties()))
        results.append(("PyTorch Version", test_pytorch_version()))
        results.append(("Core Dependencies", test_core_dependencies()))
        results.append(("pytorch3d", test_pytorch3d()))
        results.append(("nvdiffrast", test_nvdiffrast()))
        results.append(("simple-knn", test_simple_knn()))
        results.append(("RTX Optimizations", test_rtx_optimizations()))
        results.append(("Memory Test", test_memory()))
        results.append(("Tensor Operations", test_tensor_operations()))
    
    # Summary
    print("\n" + "="*60)
    print("📊 Verification Summary")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ Installation verified! Ready for training.")
        print("\n💡 Active optimizations:")
        print("  ✓ TF32 acceleration")
        print("  ✓ cuDNN auto-tuning")
        print("  ✓ Lazy image loading")
        print("  ✓ KNN caching")
        print("\n📖 Next steps:")
        print("  1. Prepare your dataset (COLMAP or NeRF format)")
        print("  2. See DOCS/MIPS_TRAIN.MD for training guide")
        print("  3. See DOCS/SUGAR_USAGE.MD for mesh extraction")
    else:
        print("⚠️  Some tests failed. Check output above.")
        print("\n🔍 Common fixes:")
        print("  1. PyTorch too old → pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130")
        print("  2. Missing pytorch3d → cd pytorch3d && pip install --no-build-isolation -e .")
        print("  3. Missing nvdiffrast → cd nvdiffrast && pip install --no-build-isolation .")
        print("  4. Missing simple-knn → cd mip-splatting/submodules/simple-knn && pip install -e .")
    print("="*60 + "\n")
    
    sys.exit(0 if all_passed else 1)

if __name__ == '__main__':
    main()
