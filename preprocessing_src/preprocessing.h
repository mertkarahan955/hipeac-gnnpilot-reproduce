#ifndef PREPROCESSING_H__
#define PREPROCESSING_H__

#include <cuda_runtime.h>
#include <vector>

#define WARP_SIZE 32
#define BLOCK_NUM 160
#define WARP_PER_BLOCK 4
#define NG_SIZE 32  // Neighbor group size

// Row panel structure - similar to warp_info
typedef struct row_panel_ {
    int row_st;
    int row_ed;
    int col_st;
    int col_ed;
    
    __host__ __device__ row_panel_() : row_st(0), row_ed(0), col_st(-1), col_ed(-1) {}
    __host__ __device__ row_panel_(int rs, int re, int cs, int ce) 
        : row_st(rs), row_ed(re), col_st(cs), col_ed(ce) {}
} row_panel;

// Neighbor group structure for ngd parallelization
typedef struct neighbor_group_ {
    int row_st;
    int col_st;
    
    __host__ __device__ neighbor_group_() : row_st(0), col_st(0) {}
    __host__ __device__ neighbor_group_(int rs, int cs) : row_st(rs), col_st(cs) {}
} neighbor_group;

// Main info structure that holds all preprocessing data
typedef struct kg_info_ {
    // Row panel info (for hetero+ kernels)
    row_panel* rp_info;
    int* rp_n;
    int rp_n_host;
    
    // Edge panel info (for edge-parallel kernels)
    row_panel* ep_info;
    int* ep_n;
    int ep_n_host;
    
    // Neighbor group info (for ngd kernels)
    neighbor_group* ng_info;
    int* ng_n;
    int ng_n_host;
    
    kg_info_() : rp_info(nullptr), rp_n(nullptr), rp_n_host(0),
                 ep_info(nullptr), ep_n(nullptr), ep_n_host(0),
                 ng_info(nullptr), ng_n(nullptr), ng_n_host(0) {}
} kg_info;

// CUDA function declaration
int64_t preprocessing_cuda(int m, int nnz, int *RowPtr, int *ColIdx, bool long_dynamic);

#endif

