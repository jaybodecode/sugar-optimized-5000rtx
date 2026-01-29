/*
 * Memory-Optimized Configuration for RTX 5060 Ti (16GB)
 * Prioritizes memory efficiency over raw speed
 * 
 * Changes from default:
 * - Smaller block size (better occupancy, less shared mem per block)
 * - More blocks can be resident simultaneously
 * - Better for larger scenes with many Gaussians
 */

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 3 // Default 3, RGB

// Memory-optimized: 8x8 blocks instead of 16x16
// - 4x less shared memory per block (7KB → 1.75KB)
// - 4x more blocks can be resident (better for large scenes)
// - Slightly more kernel launches but better memory usage
// - Recommended for scenes with >1M Gaussians
#define BLOCK_X 8
#define BLOCK_Y 8

// Alternative aggressive optimization: 4x4 blocks
// Uncomment for maximum memory efficiency (very large scenes)
// #define BLOCK_X 4
// #define BLOCK_Y 4

#endif
