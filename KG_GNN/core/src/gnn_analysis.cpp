#include "../include/KG_GNN.h"
#include <torch/extension.h>
#include <stdio.h>
#include <tuple>

extern void kg_csr_balance(int m, int nnz, int *RowPtr, int wsize, warp_info **winfo, int *winfo_n);
extern void kg_csr_balance2(int m, int nnz, int *RowPtr, int wsize, int alpha, warp_info **winfo, int *winfo_n);
extern void kg_csr_balance3(int m, int nnz, int *RowPtr, int wsize, int alpha, warp_info ***sinfo, int **sinfo_n);
extern void kg_csr_balance4(int m, int nnz, int *RowPtr, int wsize, warp_info ***sinfo, int **sinfo_n);
extern void kg_csr_schedule_locality(int m, int nnz, int *RowPtr, int *ColIdx, int bin_size, warp_info ***sinfo, int **sinfo_n);
void kg_csr_schedule_locality2(int m, int nnz, int *RowPtr, int *ColIdx, int wsize, warp_info ***sinfo, int **sinfo_n);
extern void kg_csr_block(int m, int nnz, int *RowPtr, block_info **binfo, int *binfo_n);

extern void bin_pack_construct(int m, int nnz, int *RowPtr, int *ColIdx, 
int bin_size, int pack_size, int bin_thresh, bin_pack **bp, int alpha);
extern void bin_pack_schedule(bin_pack *bp, bin_pack_info ***bpinfo, int **bpinfo_n, int bin_block, int alpha);
extern void bin_pack_schedule2(bin_pack *bp, bin_pack_info **bpinfo, int *bpinfo_n, int bin_block, int alpha);
extern void kg_finalize_cu(ana_info* ana);

int64_t kg_gcn_balance(
    torch::Tensor RowPtr,
    torch::Tensor ColIdx,
    int64_t wsize
)
{
    int m = RowPtr.size(0) - 1;
    int nnz = ColIdx.size(0);
    int winfo_n;
    warp_info *winfo;
    kg_csr_balance(m, nnz, RowPtr.data_ptr<int>(), wsize, &winfo, &winfo_n);
    ana_info *ret = new ana_info(winfo, winfo_n);
    return (int64_t)ret;
}

int64_t kg_gcn_balance2(
    torch::Tensor RowPtr,
    torch::Tensor ColIdx,
    int64_t wsize,
    int64_t alpha
)
{
    int m = RowPtr.size(0) - 1;
    int nnz = ColIdx.size(0);
    int winfo_n;
    warp_info *winfo;
    if (wsize < alpha)
    {
        printf("Wrong parameters (wsize < alpha). Set wsize = alpha + 1.\n");
        wsize = alpha + 1;
    }
    kg_csr_balance2(m, nnz, RowPtr.data_ptr<int>(), wsize, alpha, &winfo, &winfo_n);
    ana_info *ret = new ana_info(winfo, winfo_n);
    return (int64_t)ret;
}

int64_t kg_gcn_balance3(
    torch::Tensor RowPtr,
    torch::Tensor ColIdx,
    int64_t wsize,
    int64_t alpha
)
{
    int m = RowPtr.size(0) - 1;
    int nnz = ColIdx.size(0);
    warp_info **sinfo;
    int *sinfo_n;
    kg_csr_balance3(m, nnz, RowPtr.data_ptr<int>(), wsize, alpha, &sinfo, &sinfo_n);
    ana_info *ret = new ana_info(sinfo, sinfo_n);
    return (int64_t)ret;
}

int64_t kg_gcn_balance4(
    torch::Tensor RowPtr,
    torch::Tensor ColIdx,
    int64_t wsize
)
{
    int m = RowPtr.size(0) - 1;
    int nnz = ColIdx.size(0);
    warp_info **sinfo;
    int *sinfo_n;
    kg_csr_balance4(m, nnz, RowPtr.data_ptr<int>(), wsize, &sinfo, &sinfo_n);
    ana_info *ret = new ana_info(sinfo, sinfo_n);
    return (int64_t)ret;
}

int64_t kg_gcn_schedule_locality(
    torch::Tensor RowPtr,
    torch::Tensor ColIdx,
    int bin_size
)
{
    int m = RowPtr.size(0) - 1;
    int nnz = ColIdx.size(0);
    warp_info **sinfo;
    int *sinfo_n;
    kg_csr_schedule_locality(m, nnz, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), bin_size, &sinfo, &sinfo_n);
    ana_info *ret = new ana_info(sinfo, sinfo_n);
    return (int64_t)ret;
}

int64_t kg_gcn_block_schedule(
    torch::Tensor RowPtr,
    torch::Tensor ColIdx
)
{
    int m = RowPtr.size(0) - 1;
    int nnz = ColIdx.size(0);

    int binfo_n;
    block_info *binfo;
    kg_csr_block(m, nnz, RowPtr.data_ptr<int>(), &binfo, &binfo_n);

    ana_info *ret = new ana_info(binfo, binfo_n);
    return (int64_t)ret;
}

int64_t kg_gcn_bin_pack(
    torch::Tensor RowPtr, 
    torch::Tensor ColIdx,
    int64_t bin_size,
    int64_t pack_size,
    int64_t bin_thresh,
    int64_t bin_block,
    int64_t wsize,
    int64_t alpha
)
{
    int m = RowPtr.size(0) - 1;
    int nnz = ColIdx.size(0);

    bin_pack *bp;

    bin_pack_construct(m, nnz, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), 
    bin_size, pack_size, bin_thresh, &bp, alpha);

    // bin_pack_info **bpinfo;
    // int *bpinfo_n;

    // bin_pack_schedule(bp, &bpinfo, &bpinfo_n, bin_block, alpha);

    bin_pack_info *bpinfo2;
    int bpinfo_n2 = 0;
    bin_pack_schedule2(bp, &bpinfo2, &bpinfo_n2, bin_block, alpha);

    warp_info *spinfo = NULL;
    int spinfo_n;

    int m_sp = bp->spn;
    int nnz_sp = bp->RowPtr_sp_h[m_sp];

    if (m_sp > 0)
    {
        kg_csr_balance2(m_sp, nnz_sp, &(bp->RowPtr_sp_h[0]), wsize, alpha, &spinfo, &spinfo_n);
    }

    //ana_info *ret = new ana_info(bp, bpinfo, bpinfo_n, spinfo, spinfo_n);
    ana_info *ret = new ana_info(bp, bpinfo2, bpinfo_n2, spinfo, spinfo_n);

    return (int64_t)ret;
}

void kg_gcn_finalize(
    int64_t ana_add
)
{
    ana_info* ptr = (ana_info*)ana_add;

    kg_finalize_cu(ptr);

    delete ptr;
}