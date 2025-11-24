#ifndef KG_GNN__
#define KG_GNN__

#include "GPU_setup.h"
#include "utils.h"
#include <stdlib.h>
#include <vector>
#include <cuda_runtime.h>

using namespace std;

typedef struct block_info_
{
    int row_st;
    int row_ed;

    block_info_() {}
    block_info_(int row_st_in, int row_ed_in): row_st(row_st_in), row_ed(row_ed_in) {}

} block_info;

#define SINGLE_PCK_THRESH 256

typedef struct block_info2_
{
    int row_st;
    int row_ed;
    int col_st = -1;
    int col_ed = -1;

    block_info2_() {}
    __host__ __device__ block_info2_(int row_st_in, int row_ed_in): row_st(row_st_in), row_ed(row_ed_in) {}
} block_info2;

typedef struct warp_info_
{
    int row_st;
    int row_ed;
    int col_st;
    int col_ed;

    warp_info_() {}
    warp_info_(int row_st_in, int row_ed_in, int col_st_in, int col_ed_in):
    row_st(row_st_in), row_ed(row_ed_in), col_st(col_st_in), col_ed(col_ed_in) {}
} warp_info;

// Special format for KG-GNN
typedef struct bin_pack_
{
    // package part
    int *PckPtr;
    int *PckCont;
    int Pckn;

    // sparse part
    int *RowPtr_sp;
    int *ColIdx_sp;
    int spn;

    // on host
    std::vector<int> PckPtr_h;
    std::vector<int> PckCont_h;

    std::vector<int> RowPtr_sp_h;
    std::vector<int> ColIdx_sp_h;

    // Auxilary array for scheduling
    std::vector<int> BinPtr_h;
    std::vector<int> BinLoad;
    std::vector<int> PckLoad;

    bin_pack_() {}
    void bin_pack_cpy(int *PckPtr_in, int *PckCont_in, int *RowPtr_sp_in, int *ColIdx_sp_in)
    {
        PckPtr = PckPtr_in;
        PckCont = PckCont_in;
        //Pckn = Pckn_in;
        RowPtr_sp = RowPtr_sp_in;
        ColIdx_sp = ColIdx_sp_in;
    }
} bin_pack;

// info for bin_pack
typedef struct bin_pack_info_
{
    int bp_st;
    int bp_ed;

    bin_pack_info_() {}
    bin_pack_info_(int bp_st_in, int bp_ed_in): bp_st(bp_st_in), bp_ed(bp_ed_in) {}

} bin_pack_info;

typedef struct ana_info_
{
    warp_info *winfo = NULL;
    int winfo_n;

    warp_info **sinfo = NULL;
    int *sinfo_n;

    block_info *binfo = NULL;
    int binfo_n;

    bin_pack *bp = NULL;
    bin_pack_info **bpinfo = NULL;
    int *bpinfo_n;
    bin_pack_info *bpinfo2 = NULL;
    int bpinfo_n2;
    warp_info *spinfo = NULL;
    int spinfo_n;

    ana_info_(block_info *binfo_in, int binfo_n_in): binfo(binfo_in), binfo_n(binfo_n_in) {}
    ana_info_(warp_info **sinfo_in, int *sinfo_n_in): sinfo(sinfo_in), sinfo_n(sinfo_n_in) {}
    ana_info_(warp_info *winfo_in, int winfo_n_in): winfo(winfo_in), winfo_n(winfo_n_in) {}
    ana_info_(bin_pack *bp_in, bin_pack_info **bpinfo_in, int *bpinfo_n_in, warp_info *spinfo_in, int spinfo_n_in):
    bp(bp_in), bpinfo(bpinfo_in), bpinfo_n(bpinfo_n_in), spinfo(spinfo_in), spinfo_n(spinfo_n_in) {}
    ana_info_(bin_pack *bp_in, bin_pack_info *bpinfo2_in, int bpinfo_n2_in, warp_info *spinfo_in, int spinfo_n_in):
    bp(bp_in), bpinfo2(bpinfo2_in), bpinfo_n2(bpinfo_n2_in), spinfo(spinfo_in), spinfo_n(spinfo_n_in) {}

} ana_info;


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
//int64_t preprocessing_cuda(int m, int nnz, int *RowPtr, int *ColIdx, bool long_dynamic);

#endif

