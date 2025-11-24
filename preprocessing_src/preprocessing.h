#pragma once

#define SINGLE_PCK_THRESH 8192
#define DP_BLOCK_SIZE 128
#define DP_NNZ_PER_BLOCK SINGLE_PCK_THRESH
#define DP_NNZ_PER_WARP (SINGLE_PCK_THRESH * WARP_SIZE / DP_BLOCK_SIZE)
#define LONG_BLOCK_NNZ (SINGLE_PCK_THRESH)
#define NG_SIZE 32

typedef struct row_panel_
{
    int row_st;
    int row_ed;
    int col_st = -1;
    int col_ed = -1;

    row_panel_() {}
    __host__ __device__ row_panel_(int row_st_in, int row_ed_in): row_st(row_st_in), row_ed(row_ed_in) {}
    __host__ __device__ row_panel_(int row_st_in, int row_ed_in, int col_st_in, int col_ed_in): 
    row_st(row_st_in), row_ed(row_ed_in), col_st(col_st_in), col_ed(col_ed_in) {}
} row_panel;

typedef struct neighbor_group_
{
    int row_st;
    int col_st = -1;
    neighbor_group_() {}
    __host__ __device__ neighbor_group_(int row_st_in, int col_st_in): row_st(row_st_in), col_st(col_st_in) {}
} neighbor_group;

typedef struct kg_info_
{
    row_panel *rp_info;
    int *rp_n;
    int rp_n_host;

    row_panel *ep_info;
    int *ep_n;
    int ep_n_host;

    neighbor_group *ng_info;
    int *ng_n;
    int ng_n_host;
    
    kg_info_() { rp_info = NULL; rp_n = NULL; ng_info = NULL; ng_n = NULL;}
} kg_info;