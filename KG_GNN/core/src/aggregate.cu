#include "../include/KG_GNN.h"
#include <cuda.h>
#include <stdio.h>
#include <cusparse.h>

__global__ void gcn_aggregate_kernel_naive(int m, int nnz, int feat_len,
int *RowPtr, int *ColIdx, float *in_feat, float *out_feat)
{
    int local_tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + local_tid;
    //int local_wid = local_tid / WARP_SIZE;
    int global_wid = global_tid / WARP_SIZE;
    int lane_id = local_tid & (WARP_SIZE - 1);

    int row = global_wid;

    //if (!lane_id) printf("warp_id %d row %d\n", global_wid, row);

    int start_ptr = RowPtr[row];
    int end_ptr = RowPtr[row + 1];
    //if (start_ptr + 32 < end_ptr) end_ptr = start_ptr + 32;
    float degree_inv = 1.0 ;/// (end_ptr - start_ptr);

    if (row < m)
    {
        int self_idx = row * feat_len;
        for (int j = lane_id; j < feat_len; j += WARP_SIZE)
        //for (int j = lane_id; j < WARP_SIZE; j+=WARP_SIZE)
        {
            float result = 0.0;
            for (int i = start_ptr; i < end_ptr; i++)
            {
                int nid = ColIdx[i];
                int feat_idx = nid * feat_len + j;
                result += in_feat[feat_idx] * degree_inv;
            }
            out_feat[self_idx + j] = result;
        }
    }
}

// Calculate a (m x k) * b (k x n) = c (m x n) and aggregate c
__global__ void gcn_aggregate_kernel_fused(int m, int nnz, int k, int n,
int *RowPtr, int *ColIdx, float *a, float* b, float *c,
block_info *binfo, int binfo_n)
{
    int bid = blockIdx.x;
    int local_tid = threadIdx.x;
    //int global_tid = blockIdx.x * BLOCK_SIZE + local_tid;
    int local_wid = local_tid / WARP_SIZE;
    //int global_wid = global_tid / WARP_SIZE;
    int lane_id = local_tid & (WARP_SIZE - 1);

    __shared__ float a_buf[M_TILE_SIZE * K_BUF_TILE];
    __shared__ float b_buf[K_TILE_SIZE * N_BUF_TILE];
    __shared__ float c_buf[M_TILE_SIZE * N_BUF_TILE];

    if (local_tid >= BLOCK_SIZE) return;

    for (int block_iter = 0; block_iter < binfo_n; block_iter++)
    {
        int block_row_st = binfo[bid * binfo_n + block_iter].row_st;
        int block_row_ed = binfo[bid * binfo_n + block_iter].row_ed;

        // if (!local_tid) printf("bid %d row_st %d row_ed %d m n k %d %d %d\n", 
        // bid, block_row_st, block_row_ed, m, n, k);

        if (block_row_st < 0) return;

        int M_TILE = kg_min(block_row_ed - block_row_st, M_TILE_SIZE);
        int N_TILE = kg_min(n, N_TILE_SIZE);
        int K_TILE = kg_min(k, K_TILE_SIZE);

        for (int n_iter = 0; n_iter < n; n_iter += N_TILE)
        {
            int n_ed = kg_min(n, n_iter + N_TILE);
            int n_len = n_ed - n_iter;

            for (int m_block = block_row_st; m_block < block_row_ed; m_block += M_TILE)
            {
                int m_ed = kg_min(block_row_ed, block_row_st + M_TILE);

                // initialize
                for (int m_iter = m_block + local_wid; m_iter < m_ed; m_iter += WARP_PER_BLOCK)
                {
                    for (int n_sub_iter = lane_id; n_sub_iter < n_len; n_sub_iter += WARP_SIZE)
                    {
                        int buf_idx = (m_iter - m_block) * N_BUF_TILE + n_sub_iter;
                        c_buf[buf_idx] = 0;
                    }
                }

                for (int k_iter = 0; k_iter < k; k_iter += K_TILE)
                {
                    int k_ed = kg_min(k, k_iter + K_TILE);
                    int k_len = k_ed - k_iter;

                    // load tiled matrix a
                    if (k <= K_TILE_SIZE)
                    {
                        int a_st = k * m_block;
                        int a_ed = k * m_ed;
                        for (int load_iter = a_st + local_tid; load_iter < a_ed; load_iter += BLOCK_SIZE)
                        {
                            int buf_idxk = (load_iter - a_st) / K_TILE;
                            int buf_idxn = (load_iter - a_st) % K_TILE;
                            a_buf[buf_idxk * K_BUF_TILE + buf_idxn] = a[load_iter];
                        }
                    }
                    else
                    {
                        for (int m_iter = m_block + local_wid; m_iter < m_ed; m_iter += WARP_PER_BLOCK)
                        {
                            int buf_idx = (m_iter - m_block) * K_BUF_TILE;
                            for (int k_sub_iter = lane_id; k_sub_iter < k_len; k_sub_iter += WARP_SIZE)
                            {
                                a_buf[buf_idx + k_sub_iter] = a[m_iter * k + k_iter + k_sub_iter];
                            }
                        }
                    }

                    // load tiled matrix b
                    if (n <= N_TILE_SIZE)
                    {
                        int b_st = n * k_iter;
                        int b_ed = n * k_ed;
                        for (int load_iter = b_st + local_tid; load_iter < b_ed; load_iter += BLOCK_SIZE)
                        {
                            int buf_idxk = (load_iter - b_st) / N_TILE;
                            int buf_idxn = (load_iter - b_st) % N_TILE;
                            b_buf[buf_idxk * N_BUF_TILE + buf_idxn] = b[load_iter];
                        }
                    }
                    else
                    {
                        // if (!bid && !local_tid && !m_block && !n_iter)
                        //     printf("this\n");
                        for (int k_sub_iter = k_iter + local_wid; k_sub_iter < k_ed; k_sub_iter += WARP_PER_BLOCK)
                        {
                            int buf_idx = (k_sub_iter - k_iter) * N_BUF_TILE;
                            for (int n_sub_iter = lane_id; n_sub_iter < n_len; n_sub_iter += WARP_SIZE)
                            {
                                // if (buf_idx + n_sub_iter == 33)
                                //     printf("(%d %d -> %d, %.3f) ", local_wid, k_sub_iter * n + n_iter + n_sub_iter, buf_idx + n_sub_iter, b[k_sub_iter * n + n_iter + n_sub_iter]);
                                // if (!bid && !m_block && !n_iter)
                                //     printf("(%d %d -> %d, %.3f) ", local_wid, k_sub_iter * n + n_iter + n_sub_iter, buf_idx + n_sub_iter, b[k_sub_iter * n + n_iter + n_sub_iter]);
                                b_buf[buf_idx + n_sub_iter] = b[k_sub_iter * n + n_iter + n_sub_iter];
                            }
                        }
                    }

                    __syncthreads();

                    // if (!bid && !local_tid && !m_block && !n_iter)
                    // {
                    //     int a_st = m * m_block;
                    //     int a_ed = k * m_ed;
                    //     printf("a_buf\n");
                    //     for (int outm = m_block; outm < m_ed; outm++)
                    //     {
                    //         for (int outk = 0; outk < K_TILE; outk++)
                    //             printf("%.3f ", a_buf[(outm - m_block) * K_BUF_TILE + outk]);
                    //         printf("\n");
                    //     }

                    //     printf("b_buf\n");
                    //     for (int outk = 0; outk < K_TILE; outk++)
                    //     {
                    //         for (int outn = 0; outn < N_TILE; outn++)
                    //             printf("(%d, %.3f)", outk * N_BUF_TILE + outn, b_buf[outk * N_BUF_TILE + outn]);
                    //         printf("\n");
                    //     }
                    // }

                    // matrix multiplication
                    #pragma unroll
                    for (int m_iter = m_block + local_wid; m_iter < m_ed; m_iter += WARP_PER_BLOCK)
                    {
                        int a_buf_idx = (m_iter - m_block) * N_BUF_TILE;
                        #pragma unroll
                        for (int n_sub_iter = lane_id; n_sub_iter < n_len; n_sub_iter += WARP_SIZE)
                        {
                            int b_buf_idx = n_sub_iter;

                            float local_sum = 0.0;

                            #pragma unroll
                            for (int k_sub_iter = 0; k_sub_iter < k_len; k_sub_iter++)
                            {
                                // local_sum = local_sum + a_buf[a_buf_idx + k_sub_iter] *
                                // b_buf[b_buf_idx + k_sub_iter * N_BUF_TILE];
                                local_sum = local_sum + a_buf[a_buf_idx] * b_buf[b_buf_idx];
                                a_buf_idx++;
                                b_buf_idx += N_BUF_TILE;
                            }

                            int c_buf_idx = (m_iter - m_block) * N_BUF_TILE + n_sub_iter;
                            c_buf[c_buf_idx] += local_sum;
                        }
                    }

                    __syncthreads();
                }

                for (int m_iter = m_block + local_wid; m_iter < m_ed; m_iter += WARP_PER_BLOCK)
                {
                    for (int n_sub_iter = lane_id; n_sub_iter < n_len; n_sub_iter += WARP_SIZE)
                    {
                        int buf_idx = (m_iter - m_block) * N_BUF_TILE + n_sub_iter;
                        int c_idx = m_iter * n + n_iter + n_sub_iter;
                        c[c_idx] = c_buf[buf_idx];
                    }
                }
            }
        }
    }
}

#define WARP_ITER_SIZE 1

__global__ void gcn_aggregate_kernel_balance(int m, int nnz, int feat_len, int feat_st, int feat_size,
int *RowPtr, int *ColIdx, float *in_feat, float *out_feat, warp_info* winfo, int winfo_n)
{
    int local_tid = threadIdx.x;
    if (local_tid >= BLOCK_SIZE) return;
    //int global_tid = blockIdx.x * blockDim.x + local_tid;
    int global_tid = blockIdx.x * BLOCK_SIZE + local_tid;
    //int local_wid = local_tid / WARP_SIZE;
    int global_wid = global_tid / WARP_SIZE;
    int lane_id = local_tid & (WARP_SIZE - 1);

    if (lane_id >= feat_size) return;

    int j = feat_st + lane_id;

    for (int tgt = global_wid * WARP_ITER_SIZE; tgt < (global_wid + 1) * WARP_ITER_SIZE; tgt++)
    {
        if (tgt > winfo_n) return;
        warp_info info = winfo[tgt];

        // int ColIdx_size = info.col_ed - info.col_st;

        // __shared__ int ColIdx_shared[WARP_PER_BLOCK][32];

        // for (int i = lane_id; i < ColIdx_size; i+=WARP_SIZE)
        //     ColIdx_shared[local_wid][i] = ColIdx[i + info.col_st];

        // #pragma unroll (1)
        for (int row = info.row_st; row < info.row_ed; row++)
        {
            int start_ptr = (RowPtr[row] > info.col_st)? RowPtr[row]: info.col_st;
            int end_ptr = (RowPtr[row + 1] < info.col_ed)? RowPtr[row + 1]: info.col_ed;
            // int start_ptr = info.col_st;
            // int end_ptr = info.col_ed;
            float degree_inv = 1.0 ;/// (RowPtr[row + 1] - RowPtr[row]);

            //if (!lane_id) printf("warp_id %d st %d ed %d\n", lane_id, start_ptr, end_ptr);

            //int atom_add = 0;
            //if (start_ptr != RowPtr[row] || end_ptr != RowPtr[row + 1]) atom_add = 1;

            int self_idx = row * feat_len;
            //int j = lane_id;
            //for (int j = lane_id; j < feat_len; j += WARP_SIZE)
            //for (int j = lane_id; j < WARP_SIZE; j+=WARP_SIZE)
            {
                float result = 0.0;
                //#pragma unroll (1)
                for (int i = start_ptr; i < end_ptr; i++)
                {
                    //int nid = ColIdx_shared[local_wid][i - info.col_st];
                    int nid = ColIdx[i];
                    //int nid = 0;
                    int feat_idx = nid * feat_len + j;
                    //feat_idx = 0;
                    result += in_feat[feat_idx] * degree_inv;
                }
                
                //if (atom_add)
                //{
                    //atomicAdd(&out_feat[0], result);
                atomicAdd(&out_feat[self_idx + j], result);
                //}
                //else
                //{
                    //out_feat[0] = result;
                //    out_feat[self_idx + j] = result;
                //}
            }
        }
    }
}

__global__ void gcn_aggregate_kernel_balance4(int m, int nnz, int feat_len, int feat_st, int feat_size,
int *RowPtr, int *ColIdx, float *in_feat, float *out_feat, warp_info* winfo, int winfo_n)
{
    int local_tid = threadIdx.x;
    if (local_tid >= BLOCK_SIZE) return;
    //int global_tid = blockIdx.x * blockDim.x + local_tid;
    int global_tid = blockIdx.x * BLOCK_SIZE + local_tid;
    //int local_wid = local_tid / WARP_SIZE;
    int global_wid = global_tid / WARP_SIZE;
    int lane_id = local_tid & (WARP_SIZE - 1);

    if (lane_id >= feat_size) return;

    int j = feat_st + lane_id;

    for (int tgt = global_wid * WARP_ITER_SIZE; tgt < (global_wid + 1) * WARP_ITER_SIZE; tgt++)
    {
        if (tgt > winfo_n) return;
        warp_info info = winfo[tgt];

        for (int row = info.row_st; row < info.row_ed; row++)
        {
            int start_ptr = (RowPtr[row] > info.col_st)? RowPtr[row]: info.col_st;
            int end_ptr = (RowPtr[row + 1] < info.col_ed)? RowPtr[row + 1]: info.col_ed;
            float degree_inv = 1.0 ;

            int self_idx = row * feat_len;

            float result[4];
            #pragma unroll
            for (int feat_iter = 0; feat_iter < 4; feat_iter++)
                result[feat_iter] = 0.0;
            
            #pragma unroll
            for (int feat_iter = 0; feat_iter < 4; feat_iter++)
            {
                #pragma unroll
                for (int i = start_ptr; i < end_ptr; i++)
                {
                    int nid = ColIdx[i];

                    //#pragma unroll

                        int feat_idx = nid * feat_len + j + feat_iter * WARP_SIZE;
                        result[feat_iter] += in_feat[feat_idx] * degree_inv;
                    
                }
            }

            #pragma unroll
            for (int feat_iter = 0; feat_iter < 4; feat_iter++)
                atomicAdd(&out_feat[self_idx + j + feat_iter * WARP_SIZE], result[feat_iter]);

        }
    }
}

__global__ void gcn_aggregate_kernel_balance42(int m, int nnz, int feat_len, int feat_st, int feat_size,
int *RowPtr, int *ColIdx, float *in_feat, float *out_feat, warp_info* winfo, int winfo_n)
{
    int local_tid = threadIdx.x;
    if (local_tid >= BLOCK_SIZE) return;
    //int global_tid = blockIdx.x * blockDim.x + local_tid;
    int global_tid = blockIdx.x * BLOCK_SIZE + local_tid;
    int local_wid = local_tid / WARP_SIZE;
    int global_wid = global_tid / WARP_SIZE;
    int lane_id = local_tid & (WARP_SIZE - 1);

    //if (lane_id >= feat_size) return;

    int j = feat_st + WARP_SIZE * blockIdx.y + lane_id;
    if (j >= feat_len) return;

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
            {
                float result = 0.0;
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

template <int FEAT_LEN>
__global__ void gcn_aggregate_kernel_balance_aligned(int m, int nnz, int feat_len, int feat_st,
int *RowPtr, int *ColIdx, float *in_feat, float *out_feat, warp_info* winfo, int winfo_n)
{
    int local_tid = threadIdx.x;
    if (local_tid >= BLOCK_SIZE) return;
    //int global_tid = blockIdx.x * blockDim.x + local_tid;
    int global_tid = blockIdx.x * BLOCK_SIZE + local_tid;
    //int local_wid = local_tid / WARP_SIZE;
    int global_wid = global_tid / WARP_SIZE;
    int lane_id = local_tid & (WARP_SIZE - 1);

    //if (lane_id >= feat_size) return;

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
                float result = 0.0;
                for (int i = start_ptr; i < end_ptr; i++)
                {
                    int nid = ColIdx[i];
                    int feat_idx = nid * feat_len + j;
                    result += in_feat[feat_idx] * degree_inv;
                }
                atomicAdd(&out_feat[self_idx + j], result);
            }
            //__syncthreads();
        }
    }
}

template <int FEAT_LEN>
__global__ void gcn_aggregate_kernel_balance_aligned(int m, int nnz, int feat_len, int feat_st,
int *RowPtr, int *ColIdx, float *in_feat, float *out_feat, float *degree, warp_info* winfo, int winfo_n)
{
    int local_tid = threadIdx.x;
    if (local_tid >= BLOCK_SIZE) return;
    //int global_tid = blockIdx.x * blockDim.x + local_tid;
    int global_tid = blockIdx.x * BLOCK_SIZE + local_tid;
    //int local_wid = local_tid / WARP_SIZE;
    int global_wid = global_tid / WARP_SIZE;
    int lane_id = local_tid & (WARP_SIZE - 1);

    //if (lane_id >= feat_size) return;

    int j_st = feat_st + FEAT_LEN * blockIdx.y + lane_id;
    if (j_st >= feat_len) return;
    int j_ed = kg_min(j_st + FEAT_LEN, feat_len);

    for (int tgt = global_wid * WARP_ITER_SIZE; tgt < (global_wid + 1) * WARP_ITER_SIZE; tgt++)
    {
        if (tgt >= winfo_n) return;
        warp_info info = winfo[tgt];

        //if (lane_id == 0) printf("global_wid %d\n", global_wid);

        for (int row = info.row_st; row < info.row_ed; row++)
        {
            int start_ptr = (RowPtr[row] > info.col_st)? RowPtr[row]: info.col_st;
            int end_ptr = (RowPtr[row + 1] < info.col_ed)? RowPtr[row + 1]: info.col_ed;
            float degree_inv = degree[row];
            int self_idx = row * feat_len;
            for (int j = j_st; j < j_ed; j += WARP_SIZE)
            {
                float result = 0.0;
                for (int i = start_ptr; i < end_ptr; i++)
                {
                    int nid = ColIdx[i];
                    int feat_idx = nid * feat_len + j;
                    result += in_feat[feat_idx] * degree_inv;
                    //if (lane_id == 0) printf("%d %d %.2f\n", row, i, in_feat[feat_idx]);
                }
                atomicAdd(&out_feat[self_idx + j], result);
            }
        }
    }
}

// for turing arch
template <int FEAT_LEN>
__global__ void gcn_aggregate_kernel_balance_aligned128(int m, int nnz, int feat_len, int feat_st,
int *RowPtr, int *ColIdx, float *in_feat, float *out_feat, warp_info* winfo, int winfo_n)
{
    int local_tid = threadIdx.x;
    if (local_tid >= BLOCK_SIZE) return;
    int global_tid = blockIdx.x * BLOCK_SIZE + local_tid;
    int global_wid = global_tid / WARP_SIZE;
    int lane_id = local_tid & (WARP_SIZE - 1);

    float4* in_feat4 = reinterpret_cast<float4*>(in_feat);
    //float4* out_feat4 = reinterpret_cast<float4*>(out_feat);

    int feat_prefix_len = FEAT_LEN * blockIdx.y;
    int j_st = feat_st + feat_prefix_len / 4 + lane_id;
    if (j_st >= feat_len) return;
    //int j_ed = kg_min(j_st + FEAT_LEN, feat_len);

    int tgt = global_wid;
    warp_info info = winfo[tgt];
    for (int row = info.row_st; row < info.row_ed; row++)
    {
        int start_ptr = (RowPtr[row] > info.col_st)? RowPtr[row]: info.col_st;
        int end_ptr = (RowPtr[row + 1] < info.col_ed)? RowPtr[row + 1]: info.col_ed;
        float degree_inv = 1.0 ;
        int self_idx = row * feat_len / 4;
        //for (int j = j_st; j < j_ed; j += WARP_SIZE)
        int j = j_st;
        {
            float4 result;
            result.x = 0.0f;
            result.y = 0.0f;
            result.z = 0.0f;
            result.w = 0.0f;
            for (int i = start_ptr; i < end_ptr; i++)
            {
                int nid = ColIdx[i];
                int feat_idx = nid * feat_len / 4 + j;
                float4 n_data = in_feat4[feat_idx];
                result.x += n_data.x * degree_inv;
                result.y += n_data.y * degree_inv;
                result.z += n_data.z * degree_inv;
                result.w += n_data.w * degree_inv;
            }
            atomicAdd(&out_feat[self_idx * 4 + feat_prefix_len + lane_id * 4], result.x);
            atomicAdd(&out_feat[self_idx * 4 + feat_prefix_len + lane_id * 4 + 1], result.y);
            atomicAdd(&out_feat[self_idx * 4 + feat_prefix_len + lane_id * 4  + 2], result.z);
            atomicAdd(&out_feat[self_idx * 4 + feat_prefix_len + lane_id * 4  + 3], result.w);
        }
    }
}

__global__ void gcn_aggregate_kernel_scheduled(int m, int nnz, int feat_len,
int *RowPtr, int *ColIdx, float *in_feat, float *out_feat, warp_info** sinfo, int* sinfo_n)
{
    int local_tid = threadIdx.x;
    if (local_tid >= BLOCK_SIZE) return;
    //int global_tid = blockIdx.x * blockDim.x + local_tid;
    int global_tid = blockIdx.x * BLOCK_SIZE + local_tid;
    //int local_wid = local_tid / WARP_SIZE;
    int global_wid = global_tid / WARP_SIZE;
    int lane_id = local_tid & (WARP_SIZE - 1);

    // if (!lane_id && sinfo_n[global_wid])
    //     printf("wid %d level %d\n", global_wid, sinfo_n[global_wid]);
    // return;

    for (int schedule_level = 0; schedule_level < sinfo_n[global_wid]; schedule_level++)
    {
        warp_info info = sinfo[global_wid][schedule_level];

        //if (!lane_id) printf("st %d ed %d\n", info.row_st, info.row_ed);
        // float degree_inv = 1.0;

        // int row_first = info.row_st;
        // int start_ptr_first = info.col_st;
        // int end_ptr_first = (row_first < info.row_ed - 1)? RowPtr[row_first + 1]: info.col_ed;
        // int self_idx_first = row_first * feat_len;
        // for (int j = lane_id; j < feat_len; j += WARP_SIZE)
        // {
        //     float result = 0.0;
        //     //#pragma unroll
        //     for (int i = start_ptr_first; i < end_ptr_first; i++)
        //     {
        //         int nid = ColIdx[i];
        //         int feat_idx = nid * feat_len + j;
        //         result = fmaf(in_feat[feat_idx], degree_inv, result);
        //     }

        //     atomicAdd(&out_feat[self_idx_first + j], result);
        // }

        for (int row = info.row_st; row < info.row_ed; row++)
        {
            float degree_inv = 1.0;
            int start_ptr = (RowPtr[row] > info.col_st)? RowPtr[row]: info.col_st;
            int end_ptr = (RowPtr[row + 1] < info.col_ed)? RowPtr[row + 1]: info.col_ed;
            // int start_ptr = RowPtr[row];
            // int end_ptr = RowPtr[row + 1];
            // int row = info.row_st;
            // int start_ptr = info.col_st;
            // int end_ptr = info.col_ed;
            /// (RowPtr[row + 1] - RowPtr[row]);

            //if (!lane_id) printf("warp_id %d st %d ed %d\n", lane_id, start_ptr, end_ptr);

            //int atom_add = 1;
            //if (start_ptr != RowPtr[row] || end_ptr != RowPtr[row + 1]) atom_add = 1;

            int self_idx = row * feat_len;
            for (int j = lane_id; j < feat_len; j += WARP_SIZE)
            {
                float result = 0.0;
                //#pragma unroll
                for (int i = start_ptr; i < end_ptr; i++)
                {
                    int nid = ColIdx[i];
                    int feat_idx = nid * feat_len + j;
                    //result = fmaf(in_feat[feat_idx], degree_inv, result);
                    result += in_feat[feat_idx] * degree_inv;
                }

                atomicAdd(&out_feat[self_idx + j], result);
            }
        }

        // int row_end = info.row_st;
        // int start_ptr_end = info.col_st;
        // int end_ptr_end = (row_end < info.row_ed - 1)? RowPtr[row_end + 1]: info.col_ed;
        // int self_idx_end = row_first * feat_len;
        // for (int j = lane_id; j < feat_len; j += WARP_SIZE)
        // {
        //     float result = 0.0;
        //     //#pragma unroll
        //     for (int i = start_ptr_first; i < end_ptr_first; i++)
        //     {
        //         int nid = ColIdx[i];
        //         int feat_idx = nid * feat_len + j;
        //         result = fmaf(in_feat[feat_idx], degree_inv, result);
        //     }

        //     atomicAdd(&out_feat[self_idx_first + j], result);
        // }
    }
}

__global__ void gcn_aggregate_kernel_bin_pack(int m, int nnz, int feat_len, int feat_st, int feat_size,
int *PckPtr, int *PckCont, float *in_feat, float *out_feat, bin_pack_info** bpinfo, int* bpinfo_n)
{
    int local_tid = threadIdx.x;
    if (local_tid >= BLOCK_SIZE) return;
    //int global_tid = blockIdx.x * blockDim.x + local_tid;
    int global_tid = blockIdx.x * BLOCK_SIZE + local_tid;
    //int local_wid = local_tid / WARP_SIZE;
    int global_wid = global_tid / WARP_SIZE;
    int lane_id = local_tid & (WARP_SIZE - 1);

    if (lane_id >= feat_size) return;
    int j = feat_st + lane_id;

    for (int schedule_level = 0; schedule_level < bpinfo_n[global_wid]; schedule_level++)
    {
        bin_pack_info info = bpinfo[global_wid][schedule_level];

        // int Cont_st = PckPtr[info.bp_st];
        // int Cont_ed = PckPtr[info.bp_ed];
        // int Cont_st = info.bp_st;
        // int Cont_ed = info.bp_ed;

        // bin_pack_debug
        //if (!lane_id) printf("warp %d level %d info.bp_st/ed %d %d Cont_st %d Cont_ed %d\n", 
        //global_wid, schedule_level, info.bp_st, info.bp_ed, Cont_st, Cont_ed);

        //if (!lane_id) printf("PckCont %d %d\n", PckCont[Cont_st], PckCont[Cont_ed - 1]);

        float degree_inv = 1.0;
        //for (int j = lane_id; j < feat_len; j += WARP_SIZE)
        {
            // old
            // float result = 0.0;
            // int row = -PckCont[Cont_st] - 1;

            // for (int i = Cont_st + 1; i < Cont_ed; i++)
            // {
            //     int tmp = PckCont[i];
            //     if (tmp >= 0)
            //     {
            //         int feat_idx = tmp * feat_len + j;
            //         result = fmaf(in_feat[feat_idx], degree_inv, result);
            //     }
            //     else
            //     {
            //         atomicAdd(&out_feat[row * feat_len + j], result);
            //         result = 0.0;
            //         row = - tmp - 1;
            //     }
            // }

            // atomicAdd(&out_feat[row * feat_len + j], result);

            // new, bin_pack_bak_0920.cu
            // int i = Cont_st;
            // while (i < Cont_ed)
            // {
            //     int row = PckCont[i];
            //     int len = PckCont[i + 1];
            //     float result = 0.0;
            //     for (int k = i + 2; k < i + 2 + len; k++)
            //     {
            //         int feat_idx = PckCont[k] * feat_len + j;
            //         result = fmaf(in_feat[feat_idx], degree_inv, result);
            //     }
            //     //out_feat[row * feat_len + j] = result;
            //     atomicAdd(&out_feat[row * feat_len + j], result);
            //     i = i + 2 + len;
            // }

            // new new
            //#pragma unroll
            for (int pck = info.bp_st; pck < info.bp_ed; pck++)
            {
                int pck_st = PckPtr[pck];
                int pck_ed = PckPtr[pck + 1];
                int row = PckCont[pck_st];
                float result = 0.0;
                for (int k = pck_st + 1; k < pck_ed; k++)
                {
                    int feat_idx = PckCont[k] * feat_len + j;
                    result = fmaf(in_feat[feat_idx], degree_inv, result);
                }
                atomicAdd(&out_feat[row * feat_len + j], result);
            }
        }
    }
}

__global__ void gcn_aggregate_kernel_bin_pack2(int m, int nnz, int feat_len, int feat_st, int feat_size,
int *PckPtr, int *PckCont, float *in_feat, float *out_feat, bin_pack_info* bpinfo)
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

    bin_pack_info info = bpinfo[global_wid];

    float degree_inv = 1.0;

    // if (lane_id == 0 && info.bp_ed > info.bp_st)
    //     printf("global wid %d st %d ed %d\n", global_wid, info.bp_st, info.bp_ed);

    for (int pck = info.bp_st; pck < info.bp_ed; pck++)
    {
        int pck_st = PckPtr[pck];
        int pck_ed = PckPtr[pck + 1];
        int row = PckCont[pck_st];
        float result = 0.0;
        for (int k = pck_st + 1; k < pck_ed; k++)
        {
            int feat_idx = PckCont[k] * feat_len + j;
            result = fmaf(in_feat[feat_idx], degree_inv, result);
        }
        atomicAdd(&out_feat[row * feat_len + j], result);
    }

}

__global__ void gcn_aggregate_kernel_bin_pack2(int m, int nnz, int feat_len, int feat_st, int feat_size,
int *PckPtr, int *PckCont, float *in_feat, float *out_feat, float *degree, bin_pack_info* bpinfo)
{
    int local_tid = threadIdx.x;
    if (local_tid >= BLOCK_SIZE) return;
    //int global_tid = blockIdx.x * blockDim.x + local_tid;
    int global_tid = blockIdx.x * BLOCK_SIZE + local_tid;
    //int local_wid = local_tid / WARP_SIZE;
    int global_wid = global_tid / WARP_SIZE;
    int lane_id = local_tid & (WARP_SIZE - 1);

    int j = feat_st + WARP_SIZE * blockIdx.y + lane_id;
    if (j >= feat_len) return;

    bin_pack_info info = bpinfo[global_wid];

    // if (lane_id == 0 && info.bp_ed > info.bp_st)
    //     printf("global wid %d st %d ed %d\n", global_wid, info.bp_st, info.bp_ed);

    for (int pck = info.bp_st; pck < info.bp_ed; pck++)
    {
        int pck_st = PckPtr[pck];
        int pck_ed = PckPtr[pck + 1];
        int row = PckCont[pck_st];
        float degree_inv = degree[row];
        float result = 0.0;
        for (int k = pck_st + 1; k < pck_ed; k++)
        {
            int feat_idx = PckCont[k] * feat_len + j;
            result = fmaf(in_feat[feat_idx], degree_inv, result);
        }
        atomicAdd(&out_feat[row * feat_len + j], result);
    }

}

__global__ void gcn_aggregate_kernel_bin_pack3(int m, int nnz, int feat_len, int feat_st,
int *PckPtr, int *PckCont, int *RowPtr, int *ColIdx, float *in_feat, float *out_feat, 
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
                int feat_idx = PckCont[k] * feat_len + j;
                result = fmaf(in_feat[feat_idx], degree_inv, result);
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

            float result = 0.0;
            for (int i = start_ptr; i < end_ptr; i++)
            {
                int nid = ColIdx[i];
                int feat_idx = nid * feat_len + j;
                result += in_feat[feat_idx] * degree_inv;
                //if (lane_id == 0) printf("%d %d %.2f\n", row, i, in_feat[feat_idx]);
            }
            atomicAdd(&out_feat[self_idx + j], result);
        }
    }

}

__global__ void gcn_aggregate_kernel_balance_shared(int m, int nnz, int feat_len,
int *RowPtr, int *ColIdx, float *in_feat, float *out_feat, warp_info* winfo, int winfo_n)
{
    int local_tid = threadIdx.x;
    if (local_tid >= BLOCK_SIZE) return;
    //int global_tid = blockIdx.x * blockDim.x + local_tid;
    int global_tid = blockIdx.x * BLOCK_SIZE + local_tid;
    int local_wid = local_tid / WARP_SIZE;
    int global_wid = global_tid / WARP_SIZE;
    int lane_id = local_tid & (WARP_SIZE - 1);

    int tgt = global_wid;
    if (tgt > winfo_n) return;
    warp_info info = winfo[tgt];

    // Store embeddings of the whole thread block in shared memory to 
    int block_st = winfo[blockIdx.x * WARP_PER_BLOCK].row_st;
    int block_ed = winfo[kg_min((blockIdx.x + 1) * WARP_PER_BLOCK - 1, winfo_n)].row_ed;

    __shared__ float inner_embedding[WARP_PER_BLOCK][SHARED_EMBEDDING_SIZE];

    // if (block_ed - block_st > WARP_PER_BLOCK)
    // {
    //     printf("Error!\n");
    //     return;
    // }

    __syncthreads();

    for (int row = info.row_st; row < info.row_ed; row++)
    {
        int start_ptr = (RowPtr[row] > info.col_st)? RowPtr[row]: info.col_st;
        int end_ptr = (RowPtr[row + 1] < info.col_ed)? RowPtr[row + 1]: info.col_ed;
        float degree_inv = 1.0 / (RowPtr[row + 1] - RowPtr[row]);

        //if (!lane_id) printf("warp_id %d st %d ed %d\n", lane_id, start_ptr, end_ptr);

        int atom_add = 0;
        if (start_ptr != RowPtr[row] || end_ptr != RowPtr[row + 1]) atom_add = 1;

        int self_idx = row * feat_len;
        for (int j = lane_id; j < feat_len; j += WARP_SIZE)
        //for (int j = lane_id; j < WARP_SIZE; j+=WARP_SIZE)
        {
            for (int row = block_st + local_wid; row < block_ed; row+=WARP_SIZE)
            {
                inner_embedding[row - block_st][lane_id] = in_feat[row * feat_len + j];
            }

            __syncthreads();

            float result = 0.0;

            for (int i = start_ptr; i < end_ptr; i++)
            {
                int nid = ColIdx[i];
                float in_feat_value;
                if (nid >= block_st && nid < block_ed)
                    in_feat_value = inner_embedding[nid - block_st][lane_id];
                else
                    in_feat_value = in_feat[nid * feat_len + j];
                result += in_feat_value * degree_inv;
            }
            
            if (atom_add)
                atomicAdd(&out_feat[self_idx + j], result);
            else
                out_feat[self_idx + j] = result;
        }
    }
}

void gcn_aggregate(int m, int nnz, int feat_len,
int *RowPtr, int *ColIdx, float *in_feat, float *out_feat)
{
    // naive
    //int warp_num = m;
    int thread_num = m * WARP_SIZE;
    int block_num = (thread_num + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 grid(block_num);
    dim3 block(BLOCK_SIZE);

    gcn_aggregate_kernel_naive<<<grid, block>>>(m, nnz, feat_len,
    RowPtr, ColIdx, in_feat, out_feat);
}

void gcn_aggregate_fused(int m, int nnz, int feat_len_in, int feat_len_out,
int *RowPtr, int *ColIdx, float *in_feat, float *weight, float *out_feat,
block_info *binfo, int binfo_n)
{
    // nn + aggregation fused
    //int warp_num = m;
    //int thread_num = m * WARP_SIZE;
    //int block_num = (thread_num + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 grid(BLOCK_NUM);
    dim3 block(BLOCK_SIZE_ALIGN);

    gcn_aggregate_kernel_fused<<<grid, block>>>(m, nnz, feat_len_in, feat_len_out,
    RowPtr, ColIdx, in_feat, weight, out_feat, binfo, binfo_n);
}

void gcn_aggregate_balance(int m, int nnz, int feat_len,
int *RowPtr, int *ColIdx, float *in_feat, float *out_feat, warp_info* winfo, int winfo_n)
{
    // neighbour grouping for balance
    int warp_num = (winfo_n + WARP_ITER_SIZE - 1) / WARP_ITER_SIZE;
    int thread_num = warp_num * WARP_SIZE;
    int block_num = (thread_num + BLOCK_SIZE - 1) / BLOCK_SIZE;

    const int kernel_len = 32;
    dim3 grid(block_num, (feat_len + kernel_len - 1) / kernel_len);
    dim3 block(BLOCK_SIZE_ALIGN);
    int feat_st = 0;
    gcn_aggregate_kernel_balance_aligned<kernel_len><<<grid, block>>>(m, nnz, feat_len, feat_st,
    RowPtr, ColIdx, in_feat, out_feat, winfo, winfo_n);

    // int feat_size = WARP_SIZE;
    // dim3 grid(block_num, feat_len / WARP_SIZE);
    // dim3 block(BLOCK_SIZE_ALIGN);
    // int feat_st = 0;
    // gcn_aggregate_kernel_balance42<<<grid, block>>>(m, nnz, feat_len, feat_st, feat_size,
    // RowPtr, ColIdx, in_feat, out_feat, winfo, winfo_n);

    // const int kernel_len = 128;
    // dim3 grid(block_num, (feat_len + kernel_len - 1) / kernel_len);
    // dim3 block(BLOCK_SIZE_ALIGN);
    // int feat_st = 0;
    // gcn_aggregate_kernel_balance_aligned128<kernel_len><<<grid, block>>>(m, nnz, feat_len, feat_st,
    // RowPtr, ColIdx, in_feat, out_feat, winfo, winfo_n);
}

extern void gcn_aggregate_balance(int m, int nnz, int feat_len, int *RowPtr, int *ColIdx, 
float *in_feat, float *out_feat, float *degree, warp_info* winfo, int winfo_n)
{
    // neighbour grouping for balance
    int warp_num = (winfo_n + WARP_ITER_SIZE - 1) / WARP_ITER_SIZE;
    int thread_num = warp_num * WARP_SIZE;
    int block_num = (thread_num + BLOCK_SIZE - 1) / BLOCK_SIZE;

    const int kernel_len = 32;
    dim3 grid(block_num, (feat_len + WARP_SIZE - 1) / WARP_SIZE);
    dim3 block(BLOCK_SIZE_ALIGN);
    int feat_st = 0;
    //int feat_size = WARP_SIZE;
    gcn_aggregate_kernel_balance_aligned<kernel_len><<<grid, block>>>(m, nnz, feat_len, feat_st,
    RowPtr, ColIdx, in_feat, out_feat, degree, winfo, winfo_n);    
}

void gcn_aggregate_scheduled(int m, int nnz, int feat_len,
int *RowPtr, int *ColIdx, float *in_feat, float *out_feat, warp_info** sinfo, int* sinfo_n)
{
    //int warp_num = BLOCK_NUM * WARP_PER_BLOCK;
    //int thread_num = warp_num * WARP_SIZE;
    //int block_num = BLOCK_SIZE;

    dim3 grid(BLOCK_NUM);
    dim3 block(BLOCK_SIZE_ALIGN);

    // printf("WARP_NUM %d warp_num %d\n", WARP_NUM, warp_num);

    gcn_aggregate_kernel_scheduled<<<grid, block>>>(m, nnz, feat_len,
    RowPtr, ColIdx, in_feat, out_feat, sinfo, sinfo_n);
}

void gcn_aggregate_balance_shared(int m, int nnz, int feat_len,
int *RowPtr, int *ColIdx, float *in_feat, float *out_feat, warp_info* winfo, int winfo_n)
{
    // neighbour grouping for balance
    int warp_num = winfo_n;
    int thread_num = warp_num * WARP_SIZE;
    int block_num = (thread_num + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 grid(block_num);
    dim3 block(BLOCK_SIZE_ALIGN);

    gcn_aggregate_kernel_balance_shared<<<grid, block>>>(m, nnz, feat_len,
    RowPtr, ColIdx, in_feat, out_feat, winfo, winfo_n);
}

void gcn_aggregate_bin_pack(int m, int nnz, int feat_len,
int *PckPtr, int *PckCont, float *in_feat, float *out_feat, bin_pack_info **bpinfo, int *bpinfo_n)
{
    //int warp_num = BLOCK_NUM * WARP_PER_BLOCK;

    dim3 grid(BLOCK_NUM);
    dim3 block(BLOCK_SIZE_ALIGN);

    for (int i = 0; i < feat_len; i += WARP_SIZE)
    {
        int feat_st = i;
        int feat_size = kg_min(feat_len - i, WARP_SIZE);
        gcn_aggregate_kernel_bin_pack<<<grid, block>>>(m, nnz, feat_len, feat_st, feat_size,
        PckPtr, PckCont, in_feat, out_feat, bpinfo, bpinfo_n);
    }
}

void gcn_aggregate_bin_pack2(int m, int nnz, int feat_len,
int *PckPtr, int *PckCont, float *in_feat, float *out_feat, bin_pack_info *bpinfo, int bpinfo_n)
{
    int warp_num = (bpinfo_n + WARP_PER_BLOCK - 1) / WARP_PER_BLOCK;

    dim3 grid(warp_num);
    dim3 block(BLOCK_SIZE_ALIGN);

    for (int i = 0; i < feat_len; i += WARP_SIZE)
    {
        //printf("bpinfo_n %d\n", bpinfo_n);

        int feat_st = i;
        int feat_size = kg_min(feat_len - i, WARP_SIZE);
        gcn_aggregate_kernel_bin_pack2<<<grid, block>>>(m, nnz, feat_len, feat_st, feat_size,
        PckPtr, PckCont, in_feat, out_feat, bpinfo);
    }
}

void gcn_aggregate_bin_pack2(int m, int nnz, int feat_len,
int *PckPtr, int *PckCont, float *in_feat, float *out_feat, float *degree,
bin_pack_info *bpinfo, int bpinfo_n)
{
    int warp_num = (bpinfo_n + WARP_PER_BLOCK - 1) / WARP_PER_BLOCK;

    // dim3 grid(warp_num);
    // dim3 block(BLOCK_SIZE_ALIGN);

    int feat_size = kg_min(WARP_SIZE, feat_len);
    dim3 grid(warp_num, (feat_len + WARP_SIZE - 1) / WARP_SIZE);
    dim3 block(BLOCK_SIZE_ALIGN);
    int feat_st = 0;

    gcn_aggregate_kernel_bin_pack2<<<grid, block>>>(m, nnz, feat_len, feat_st, feat_size,
    PckPtr, PckCont, in_feat, out_feat, degree, bpinfo);
}

void gcn_aggregate_bin_pack3(int m, int nnz, int feat_len,
int *PckPtr, int *PckCont, int *RowPtr, int *ColIdx, float *in_feat, float *out_feat, 
bin_pack_info *bpinfo, int bpinfo_n, warp_info* spinfo, int spinfo_n)
{
    int block_num = (bpinfo_n + spinfo_n + WARP_PER_BLOCK - 1) / WARP_PER_BLOCK;
    // int thread_num = warp_num * WARP_SIZE;
    // int block_num = (thread_num + BLOCK_SIZE - 1) / BLOCK_SIZE;

    const int kernel_len = 32;
    dim3 grid(block_num, (feat_len + kernel_len - 1) / kernel_len);
    dim3 block(BLOCK_SIZE_ALIGN);

    gcn_aggregate_kernel_bin_pack3<<<grid, block>>>(m, nnz, feat_len, 0, PckPtr, PckCont,
    RowPtr, ColIdx, in_feat, out_feat, bpinfo, bpinfo_n, spinfo, spinfo_n);
}

float gcn_aggregate_cusparse(int m, int nnz, int feat_len,
int *RowPtr, int *ColIdx, float *Values, float *in_feat, float *out_feat, int warmup, int repetitions)
{
    cusparseHandle_t        handle;
    cusparseSpMatDescr_t    csrDescr;
    cusparseDnMatDescr_t    dnMatInputDescr, dnMatOutputDescr;
    float alpha = 1.0f, beta = 0.0f;

    CUSPARSE_CHECK(cusparseCreate(&handle));

    // creating sparse csr matrix
    CUSPARSE_CHECK(cusparseCreateCsr(&csrDescr, 
        m, m, nnz, RowPtr, ColIdx, Values, 
        CUSPARSE_INDEX_32I, // index 32-integer for indptr
        CUSPARSE_INDEX_32I, // index 32-integer for indices
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F          // datatype: 32-bit float real number
    ));

    // creating dense matrices
    CUSPARSE_CHECK(cusparseCreateDnMat(&dnMatInputDescr,
                                        m,
                                        feat_len,
                                        feat_len,
                                        in_feat,
                                        CUDA_R_32F,
                                        CUSPARSE_ORDER_ROW
    ));

    CUSPARSE_CHECK(cusparseCreateDnMat(&dnMatOutputDescr,
                                        m,
                                        feat_len,
                                        feat_len,
                                        out_feat,
                                        CUDA_R_32F,
                                        CUSPARSE_ORDER_ROW
    ));

    // allocate workspace buffer
    size_t workspace_size;
    CUSPARSE_CHECK( cusparseSpMM_bufferSize(handle,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha,
                                            csrDescr,
                                            dnMatInputDescr,
                                            &beta,
                                            dnMatOutputDescr,
                                            CUDA_R_32F,
                                            CUSPARSE_SPMM_ALG_DEFAULT,
                                            &workspace_size
                                            ));
    
    void *workspace = NULL;
    CUDA_CHECK_ERROR(cudaMalloc(&workspace, workspace_size));

    // printf("workspace size %d\n", workspace_size);

    GpuTimer gpu_timer;
    // int warmup_iter = 10;
    // int repeat_iter = 100;
    for (int iter = 0; iter < warmup + repetitions; iter++) {
        if (iter == warmup) {
            cudaDeviceSynchronize();
            gpu_timer.start();
        }

        // run SpMM
        CUSPARSE_CHECK( cusparseSpMM(handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE, // opA
                                    CUSPARSE_OPERATION_NON_TRANSPOSE, // opB
                                    &alpha,
                                    csrDescr,
                                    dnMatInputDescr, 
                                    &beta,
                                    dnMatOutputDescr,
                                    CUDA_R_32F,
                                    CUSPARSE_SPMM_ALG_DEFAULT,
                                    workspace));

        cudaDeviceSynchronize();

    }
    gpu_timer.stop();

    float kernel_dur_usecs = gpu_timer.elapsed_msecs() * 1000 / repetitions;

    CUDA_CHECK_ERROR(cudaFree(workspace));

    return kernel_dur_usecs;
}