#ifndef GPU_SETUP__
#define GPU_SETUP__

#define SM_NUM 80
#define BLOCK_NUM 160
#define WARP_SIZE 32
#define BLOCK_SIZE 128
#define BLOCK_SIZE_ALIGN 128

#define THREAD_NUM (BLOCK_NUM * BLOCK_SIZE)
#define WARP_NUM (THREAD_NUM / WARP_SIZE)
#define WARP_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)

#define SHARED_EMBEDDING_SIZE 32

// Tile size for fused kernel
#define M_TILE_SIZE 96
#define N_TILE_SIZE 32
#define K_TILE_SIZE 32

#define N_BUF_TILE 32
#define K_BUF_TILE 32

#endif
