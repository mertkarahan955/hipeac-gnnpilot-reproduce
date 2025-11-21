#include "../include/KG_GNN.h"
#include <cuda.h>
#include <stdio.h>
#include <cusparse.h>
#include <sys/time.h>

#define duration(a, b) (1.0 * (b.tv_usec - a.tv_usec + (b.tv_sec - a.tv_sec) * 1.0e6))

#define WARP_ITER_SIZE 1

__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ static float atomicMin(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

#define REGULAR 1000.0

// in_feat: node features
// a_vec: matrix of size m * 2, [a1 | a2] * [row_node_feat | col_node_feat], (i, 1) the a1 part, (i, 2) the a2 part
// max_vec: maximum exponent value of each row
// After this kernel, must calculate in_feat / max_vec

template <int FEAT_LEN>
__global__ void gat_aggregate_kernel_balance_aligned(int m, int nnz, int feat_len, int feat_st,
int *RowPtr, int *ColIdx, float *in_feat, float relu_l, float *out_feat, float *edge_weight, float *sum_vec, warp_info* winfo, int winfo_n)
{
    int local_tid = threadIdx.x;
    if (local_tid >= BLOCK_SIZE) return;
    int local_wid = local_tid / WARP_SIZE;
    int global_tid = blockIdx.x * BLOCK_SIZE + local_tid;
    int global_wid = global_tid / WARP_SIZE;
    int lane_id = local_tid & (WARP_SIZE - 1);

    int j_st = feat_st + FEAT_LEN * blockIdx.y;
    if (j_st + lane_id >= feat_len) return;
    int j_ed = kg_min(j_st + FEAT_LEN, feat_len);

    for (int tgt = global_wid * WARP_ITER_SIZE; tgt < (global_wid + 1) * WARP_ITER_SIZE; tgt++)
    {
        if (tgt >= winfo_n) return;
        warp_info info = winfo[tgt];

        // if (!lane_id && info.row_st <= 100) printf("row %d %d\n", info.row_st, info.row_ed);

        for (int row = info.row_st; row < info.row_ed; row++)
        {
            int start_ptr = (RowPtr[row] > info.col_st)? RowPtr[row]: info.col_st;
            int end_ptr = (RowPtr[row + 1] < info.col_ed)? RowPtr[row + 1]: info.col_ed;
            int self_idx = row * feat_len;

            // float e_left = a_vec[row << 1];
            // float e_sum = 0;

            for (int j = j_st + lane_id; j < j_ed; j += WARP_SIZE)
            {
                float result = 0.0;
                for (int i = start_ptr; i < end_ptr; i++)
                {
                    int nid = ColIdx[i];

                    // float e_right = a_vec[(nid << 1) + 1];
                    // float e = e_left + e_right;
                    // float e_adjust = kg_max(e, e * relu_l) - (max_vec[row] - REGULAR);
                    // //if (e_adjust > 2.0 && !lane_id) printf("row %d error %.2f nid %d\n", row, e, nid);
                    // e = expf(e_adjust);
                    
                    // e_sum += e;

                    int feat_idx = nid * feat_len + j;
                    result += in_feat[feat_idx] * edge_weight[i];

                }
                atomicAdd(&out_feat[self_idx + j], result);
            }
            // if (lane_id == 0)
            // {
            //     atomicAdd(&sum_vec[row], (float)e_sum);
            //     //if (row == 44841) printf("row %d e_sum %.4f e_max %.4f\n", row, sum_vec[row], max_vec[row]);
            // }
        }
    }
}

template <int FEAT_LEN>
__global__ void gat_aggregate_kernel_bin_pack3(int m, int nnz, int feat_len, int feat_st,
int *PckPtr, int *PckCont, int *RowPtr, int *ColIdx, float *in_feat, float *a_vec, float relu_l, float *out_feat, float *max_vec, 
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

    int j_st = feat_st + FEAT_LEN * blockIdx.y;
    if (j_st + lane_id >= feat_len) return;
    int j_ed = kg_min(j_st + FEAT_LEN, feat_len);

    if (global_wid < bpinfo_n)
    {
        bin_pack_info info = bpinfo[global_wid];

        float degree_inv = 1.0;

        for (int pck = info.bp_st; pck < info.bp_ed; pck++)
        {
            int pck_st = PckPtr[pck];
            int pck_ed = PckPtr[pck + 1];
            int row = PckCont[pck_st];
            int self_idx = row * feat_len;

            float e_left = a_vec[row << 1];
            float e_max = 0;

            for (int j = j_st + lane_id; j < j_ed; j += WARP_SIZE)
            {
                float result = 0.0;
                for (int k = pck_st + 1; k < pck_ed; k++)
                {
                    int nid = PckCont[k];

                    float e_right = a_vec[(nid << 1) + 1];
                    float e = e_left + e_right;

                    e = __expf(kg_max(e, e * relu_l));
                    if (j == j_st) e_max += e;

                    int feat_idx = nid * feat_len + j;
                    result += in_feat[feat_idx] * e;

                    // int feat_idx = PckCont[k] * feat_len + j;
                    // result = fmaf(in_feat[feat_idx], degree_inv, result);
                }

                atomicAdd(&out_feat[self_idx + j], result);
            }
            if (lane_id == 0)
            {
                atomicAdd(&max_vec[row], (float)e_max);
            }

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

            float e_left = a_vec[row << 1];
            float e_max = 0;

            for (int j = j_st + lane_id; j < j_ed; j += WARP_SIZE)
            {
                float result = 0.0;
                for (int i = start_ptr; i < end_ptr; i++)
                {
                    int nid = ColIdx[i];

                    float e_right = a_vec[(nid << 1) + 1];
                    float e = e_left + e_right;
                    e = __expf(kg_max(e, e * relu_l));
                    if (j == j_st) e_max += e;

                    int feat_idx = nid * feat_len + j;
                    result += in_feat[feat_idx] * e;

                }
                atomicAdd(&out_feat[self_idx + j], result);
            }
            if (lane_id == 0)
            {
                //printf("row %d e_max %.4f\n", row, e_max);
                atomicAdd(&max_vec[row], (float)e_max);
            }
        }
    }

}

__global__ void gat_aggregate_kernel_get_sum(int m, int nnz, 
int *RowPtr, int *ColIdx, float relu_l, float *a_vec, float *sum_vec, float *edge_weight)
{
    int local_tid = threadIdx.x;
    int local_wid = local_tid / WARP_SIZE;
    int global_tid = blockIdx.x * blockDim.x + local_tid;
    int global_wid = global_tid / WARP_SIZE;
    int lane_id = local_tid & (WARP_SIZE - 1);

    if (global_wid >= m) return;

    int row = global_wid;
    int start_ptr = RowPtr[row];
    int end_ptr = RowPtr[row + 1];

    float e_left = a_vec[row << 1];
    float e_max = 0;

    for (int i = start_ptr + lane_id; i < end_ptr; i += WARP_SIZE)
    {
        int nid = ColIdx[i];
        float e_right = a_vec[(nid << 1) + 1];
        float e = e_left + e_right;
        e_max = kg_max(kg_max(e, e * relu_l) + REGULAR, e_max);
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        float e_comm = __shfl_down_sync(0xffffffff, e_max, offset);
        e_max = kg_max(e_comm, e_max);
    }

    e_max = __shfl_sync(0xffffffff, e_max, 0) - REGULAR;

    float e_sum = 0;

    for (int i = start_ptr + lane_id; i < end_ptr; i += WARP_SIZE)
    {
        int nid = ColIdx[i];
        float e_right = a_vec[(nid << 1) + 1];
        float e = e_left + e_right;
        float e_adjust = __expf(kg_max(e, e * relu_l) - e_max);

        e_sum += e_adjust;

        edge_weight[i] = e_adjust;
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        float e_comm = __shfl_down_sync(0xffffffff, e_sum, offset);
        e_sum += e_comm;
    }

    if (lane_id == 0)
    {
        sum_vec[row] = e_sum;
        //printf("row %d max %.3f sum %.3f\n", row, e_max, e_sum);
    }

}

__global__ void gat_divide_sum(int m, int feat_len, float *out_feat, float *sum_vec)
{
    int local_tid = threadIdx.x;
    int local_wid = local_tid / WARP_SIZE;
    int global_tid = blockIdx.x * blockDim.x + local_tid;
    int global_wid = global_tid / WARP_SIZE;
    int lane_id = local_tid & (WARP_SIZE - 1);

    if (global_wid >= m) return;

    for (int j = lane_id; j < feat_len; j+=WARP_SIZE)
    {
        out_feat[global_wid * feat_len + j] = out_feat[global_wid * feat_len + j] / sum_vec[global_wid];
        // if (isnan(out_feat[global_wid * feat_len + j]))
        // {
        //     if (!j)
        //         printf("error at row %d feat %d %.3f\n", global_wid, j, sum_vec[global_wid]);
        // }
    }
}

void gat_aggregate_balance(int m, int nnz, int feat_len, int *RowPtr, int *ColIdx, 
float *in_feat, float *a_vec, float relu_l, float *out_feat, float *sum_vec, float *edge_weight,
warp_info* winfo, int winfo_n)
{
    int warp_num = m;
    const int divide_bs = 128;
    int thread_num = warp_num * WARP_SIZE;
    int block_num = (thread_num + divide_bs - 1) / divide_bs;

    // printf("--GAT time--\n");

    cudaDeviceSynchronize();

    struct timeval tv_begin, tv_end;
    gettimeofday(&tv_begin, NULL);

    gat_aggregate_kernel_get_sum<<<block_num, divide_bs>>>(m, nnz, 
    RowPtr, ColIdx, relu_l, a_vec, sum_vec, edge_weight);
    cudaDeviceSynchronize();

    gettimeofday(&tv_end, NULL);

    // printf("Attention time: %.2f us\n", duration(tv_begin, tv_end));

    // printf("???\n");
    warp_num = winfo_n;
    thread_num = warp_num * WARP_SIZE;
    block_num = (thread_num + BLOCK_SIZE - 1) / BLOCK_SIZE;

    const int kernel_len = 32;
    dim3 grid(block_num, (feat_len + kernel_len - 1) / kernel_len);
    dim3 block(BLOCK_SIZE_ALIGN);

    // printf("%d %d %d\n", winfo_n, block_num, (feat_len + kernel_len - 1) / kernel_len);

    gettimeofday(&tv_begin, NULL);

    gat_aggregate_kernel_balance_aligned<kernel_len><<<grid, block>>>
    (m, nnz, feat_len, 0, RowPtr, ColIdx, in_feat, relu_l, out_feat, edge_weight, sum_vec, winfo, winfo_n);
    cudaDeviceSynchronize();

    gettimeofday(&tv_end, NULL);

    // printf("Aggregation time: %.2f us\n", duration(tv_begin, tv_end));

    warp_num = m;
    block_num = (warp_num * WARP_SIZE + divide_bs - 1) / divide_bs;

    gettimeofday(&tv_begin, NULL);

    gat_divide_sum<<<block_num, divide_bs>>>(m, feat_len, out_feat, sum_vec);
    cudaDeviceSynchronize();

    gettimeofday(&tv_end, NULL);

    // printf("Divid sum time: %.2f us\n", duration(tv_begin, tv_end));
    // printf("------------\n");
}

void gat_aggregate_bin_pack3(int m, int nnz, int feat_len, int *RowPtr, int *ColIdx, int *PckPtr, int *PckCont, 
float *in_feat, float *a_vec, float relu_l, float *out_feat, float *max_vec,
bin_pack_info *bpinfo, int bpinfo_n, warp_info* spinfo, int spinfo_n)
{
    int block_num = (bpinfo_n + spinfo_n + WARP_PER_BLOCK - 1) / WARP_PER_BLOCK;
    // int thread_num = warp_num * WARP_SIZE;
    // int block_num = (thread_num + BLOCK_SIZE - 1) / BLOCK_SIZE;

    const int kernel_len = 32;
    dim3 grid(block_num, (feat_len + kernel_len - 1) / kernel_len);
    dim3 block(BLOCK_SIZE_ALIGN);

    gat_aggregate_kernel_bin_pack3<kernel_len><<<grid, block>>>(m, nnz, feat_len, 0, PckPtr, PckCont,
    RowPtr, ColIdx, in_feat, a_vec, relu_l, out_feat, max_vec, bpinfo, bpinfo_n, spinfo, spinfo_n);

    // const int divide_bs = 128;
    // int warp_num = m;
    // block_num = (warp_num + divide_bs - 1) / divide_bs;
    // block = dim3(divide_bs);

    // gat_divide_max<<<grid, block>>>(m, feat_len, out_feat, max_vec);

    // cudaDeviceSynchronize();
}
