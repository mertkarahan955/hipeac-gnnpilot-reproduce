#include "../include/KG_GNN.h"
#include <cuda.h>
#include <stdio.h>
#include <cusparse.h>

#define WARP_ITER_SIZE 1

template <int FEAT_LEN>
__global__ void gin_aggregate_kernel_balance_aligned(int m, int nnz, int feat_len, int feat_st,
int *RowPtr, int *ColIdx, float *in_feat, float *out_feat, float eps, warp_info* winfo, int winfo_n)
{
    int local_tid = threadIdx.x;
    if (local_tid >= BLOCK_SIZE) return;
    int global_tid = blockIdx.x * BLOCK_SIZE + local_tid;
    int global_wid = global_tid / WARP_SIZE;
    int lane_id = local_tid & (WARP_SIZE - 1);

    int j_st = feat_st + FEAT_LEN * blockIdx.y + lane_id;
    if (j_st >= feat_len) return;
    int j_ed = kg_min(j_st + FEAT_LEN, feat_len);

    for (int tgt = global_wid * WARP_ITER_SIZE; tgt < (global_wid + 1) * WARP_ITER_SIZE; tgt++)
    {
        if (tgt >= winfo_n) return;
        warp_info info = winfo[tgt];
        for (int row = info.row_st; row < info.row_ed; row++)
        {
            int start_ptr = (RowPtr[row] > info.col_st)? RowPtr[row]: info.col_st;
            int end_ptr = (RowPtr[row + 1] < info.col_ed)? RowPtr[row + 1]: info.col_ed;
            float degree_inv = 1.0 ;
            int self_idx = row * feat_len;

            for (int j = j_st; j < j_ed; j += WARP_SIZE)
            {
                float init_val = 0.0;
                if (start_ptr == RowPtr[row])
                    init_val = in_feat[self_idx + j] * (1.0 + eps);
                float result = init_val;
                for (int i = start_ptr; i < end_ptr; i++)
                {
                    int nid = ColIdx[i];
                    int feat_idx = nid * feat_len + j;
                    result += in_feat[feat_idx] * degree_inv;
                }
                atomicAdd(&out_feat[self_idx + j], result);
            }
        }
    }
}

__global__ void gin_aggregate_kernel_bin_pack3(int m, int nnz, int feat_len, int feat_st,
int *PckPtr, int *PckCont, int *RowPtr, int *ColIdx, float *in_feat, float *out_feat, float eps,
bin_pack_info *bpinfo, int bpinfo_n, warp_info* spinfo, int spinfo_n)
{
    int local_tid = threadIdx.x;
    if (local_tid >= BLOCK_SIZE) return;
    //int global_tid = blockIdx.x * blockDim.x + local_tid;
    int global_tid = blockIdx.x * BLOCK_SIZE + local_tid;
    //int local_wid = local_tid / WARP_SIZE;
    int global_wid = global_tid / WARP_SIZE;
    int lane_id = local_tid & (WARP_SIZE - 1);

    //if (lane_id >= feat_size) return;

    int j = feat_st + WARP_SIZE * blockIdx.y + lane_id;
    if (j >= feat_len) return;

    if (global_wid < bpinfo_n)
    {
        bin_pack_info info = bpinfo[global_wid];

        float degree_inv = 1.0;

        for (int pck = info.bp_st; pck < info.bp_ed; pck++)
        {
            int pck_st = PckPtr[pck];
            int pck_ed = PckPtr[pck + 1];
            int row = PckCont[pck_st];
            float result = 0.0;

            for (int k = pck_st + 1; k < pck_ed; k++)
            {
                int nid = PckCont[k];
                int feat_idx = nid * feat_len + j;
                result = fmaf(in_feat[feat_idx], degree_inv, result);
                if (nid == row)
                    result = fmaf(in_feat[feat_idx], 1.0 + eps, result);
            }
            atomicAdd(&out_feat[row * feat_len + j], result);
        }
    }
    else if (global_wid < bpinfo_n + spinfo_n)
    {
        warp_info info = spinfo[global_wid - bpinfo_n];

        for (int row = info.row_st; row < info.row_ed; row++)
        {
            int start_ptr = (RowPtr[row] > info.col_st)? RowPtr[row]: info.col_st;
            int end_ptr = (RowPtr[row + 1] < info.col_ed)? RowPtr[row + 1]: info.col_ed;
            
            float degree_inv = 1.0;//degree[row];
            int self_idx = row * feat_len;
            // float init_val = 0.0;
            // if (start_ptr == RowPtr[row])
            //     init_val = in_feat[self_idx + j] * (1.0 + eps);

            float result = 0.0;
            for (int i = start_ptr; i < end_ptr; i++)
            {
                int nid = ColIdx[i];
                int feat_idx = nid * feat_len + j;
                result += in_feat[feat_idx] * degree_inv;
                if (nid == row)
                    result = fmaf(in_feat[feat_idx], 1.0 + eps, result);
            }
            atomicAdd(&out_feat[self_idx + j], result);
        }
    }

}

__global__ void gin_plus_self(int m, int feat_len, float *in_feat, float *out_feat, float eps)
{
    int local_tid = threadIdx.x;
    int local_wid = local_tid / WARP_SIZE;
    int global_tid = blockIdx.x * blockDim.x + local_tid;
    int global_wid = global_tid / WARP_SIZE;
    int lane_id = local_tid & (WARP_SIZE - 1);
    int j = threadIdx.y * WARP_SIZE + lane_id;

    if (global_wid >= m) return;
    if (j >= feat_len) return;

    //if (global_wid == m -1 && !lane_id) printf("???\n");

    // for (int j = lane_id; j < feat_len; j+=WARP_SIZE)
    // {
        out_feat[global_wid * feat_len + j] = out_feat[global_wid * feat_len + j] + in_feat[global_wid * feat_len + j] * (1.0 + eps);
    // }
}

void gin_aggregate_balance(int m, int nnz, int feat_len,
int *RowPtr, int *ColIdx, float *in_feat, float *out_feat, float eps, warp_info* winfo, int winfo_n)
{
    // neighbour grouping for balance
    int warp_num = (winfo_n + WARP_ITER_SIZE - 1) / WARP_ITER_SIZE;
    int thread_num = warp_num * WARP_SIZE;
    int block_num = (thread_num + BLOCK_SIZE - 1) / BLOCK_SIZE;

    const int kernel_len = 32;
    dim3 grid(block_num, (feat_len + kernel_len - 1) / kernel_len);
    dim3 block(BLOCK_SIZE_ALIGN);
    int feat_st = 0;
    gin_aggregate_kernel_balance_aligned<kernel_len><<<grid, block>>>(m, nnz, feat_len, feat_st,
    RowPtr, ColIdx, in_feat, out_feat, eps, winfo, winfo_n);

    // const int divide_bs = 128;
    // warp_num = m;
    // grid = (warp_num * WARP_SIZE + divide_bs - 1) / divide_bs;
    // block = dim3(divide_bs, (feat_len + WARP_SIZE - 1) / WARP_SIZE);

    // gin_plus_self<<<grid, block>>>(m, feat_len, in_feat, out_feat, eps);

    cudaDeviceSynchronize();
}

extern __global__ void gcn_aggregate_kernel_bin_pack3(int m, int nnz, int feat_len, int feat_st,
int *PckPtr, int *PckCont, int *RowPtr, int *ColIdx, float *in_feat, float *out_feat, 
bin_pack_info *bpinfo, int bpinfo_n, warp_info* spinfo, int spinfo_n);

void gin_aggregate_bin_pack3(int m, int nnz, int feat_len,
int *PckPtr, int *PckCont, int *RowPtr, int *ColIdx, float *in_feat, float *out_feat, float eps,
bin_pack_info *bpinfo, int bpinfo_n, warp_info* spinfo, int spinfo_n)
{
    int block_num = (bpinfo_n + spinfo_n + WARP_PER_BLOCK - 1) / WARP_PER_BLOCK;

    const int kernel_len = 32;
    dim3 grid(block_num, (feat_len + kernel_len - 1) / kernel_len);
    dim3 block(BLOCK_SIZE_ALIGN);

    gin_aggregate_kernel_bin_pack3<<<grid, block>>>(m, nnz, feat_len, 0, PckPtr, PckCont,
    RowPtr, ColIdx, in_feat, out_feat, eps, bpinfo, bpinfo_n, spinfo, spinfo_n);

    // gcn_aggregate_kernel_bin_pack3<<<grid, block>>>(m, nnz, feat_len, 0, PckPtr, PckCont,
    // RowPtr, ColIdx, in_feat, out_feat, bpinfo, bpinfo_n, spinfo, spinfo_n);

    // const int divide_bs = 128;
    // int warp_num = m;
    // grid = (warp_num * WARP_SIZE + divide_bs - 1) / divide_bs;
    // block = dim3(divide_bs, (feat_len + WARP_SIZE - 1) / WARP_SIZE);

    // gin_plus_self<<<grid, block>>>(m, feat_len, in_feat, out_feat, eps);

    cudaDeviceSynchronize();
}