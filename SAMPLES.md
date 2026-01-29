# Mip-NeRF 360 Dataset Samples

## Dataset Information

This project uses sample scenes from the **Mip-NeRF 360** dataset for testing and benchmarks. Due to GitHub's file size limits, the sample datasets are **not included** in this repository.

**You must download them separately:**

**Source:** https://jonbarron.info/mipnerf360/  
**Direct Download:** http://storage.googleapis.com/gresearch/refraw360/360_v2.zip (18GB)

## Dataset Citation

If you use this dataset, please cite:

```bibtex
@inproceedings{barron2022mipnerf360,
  title={Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields},
  author={Barron, Jonathan T. and Mildenhall, Ben and Verbin, Dor and Srinivasan, Pratul P. and Hedman, Peter},
  booktitle={CVPR},
  year={2022}
}
```

**Reference:**  
Barron, Jonathan T. et al. "Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields." CVPR, 2022.

## License

This dataset is released under Google's standard research dataset license.  
Please refer to the original source for full terms and conditions.

## Scenes Included

- **bicycle/** - Outdoor bicycle scene
- **garden/** - Outdoor garden scene with vegetation

## Dataset Characteristics

**Resolution:** Original images up to 5187×3361 pixels  
**Format:** JPEG images with COLMAP camera poses  
**Scene Type:** Outdoor unbounded 360° capture  
**Images per scene:** ~150-200 images

## Setup Instructions

1. **Download the dataset:**
   ```bash
   wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
   unzip 360_v2.zip
   ```

2. **Extract scenes to SAMPLES directory:**
   ```bash
   mkdir -p SAMPLES
   mv 360_v2/bicycle SAMPLES/
   mv 360_v2/garden SAMPLES/
   ```

3. **Train with sample datasets:**
   ```bash
   # Mip-Splatting
   cd mip-splatting
   python train.py -s ../SAMPLES/garden --iteration 7000 -r 2
   
   # SuGaR
   cd ../SuGaR
   python train.py -s ../SAMPLES/garden -c ../path/to/mip-output -i 7000
   ```

## Usage in This Project

These scenes are used for:
1. **Mip-splatting training** - Multi-resolution 3D Gaussian Splatting
2. **SuGaR training** - Extracting textured meshes from trained Gaussians
3. **Baseline comparison** - Validating optimizations and quality metrics

See [DOCS/MIPS_TRAIN.MD](DOCS/MIPS_TRAIN.MD) and [DOCS/SUGAR_USAGE.MD](DOCS/SUGAR_USAGE.MD) for training instructions.

## Performance Benchmarks

Training on **garden scene** (RTX 5060 Ti 16GB):
- Mip-splatting (7K iter): ~30 minutes, PSNR 28.48 dB
- SuGaR coarse (15K iter): ~13 min per 200 iterations with optimizations
- Final Gaussians: ~5M points
- Memory usage: 15.73GB VRAM (with SuGaR optimizations)

**Note:** Sample datasets are excluded from this repository due to size constraints (3.7GB). You can use your own datasets or download the official Mip-NeRF 360 dataset as shown above.
