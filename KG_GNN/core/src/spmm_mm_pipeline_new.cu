#include "../include/KG_GNN.h"
#include <cuda.h>
#include <stdio.h>
#include "cublas_v2.h"
#include <cuda_pipeline.h>
// #include <cooperative_groups.h>
// #include <cooperative_groups/memcpy_async.h>

// namespace cg = cooperative_groups;

#define MM_MEM_PER_BLOCK 4
#define MM_CALC_PER_BLOCK 4
#define SPMM_PER_BLOCK 16

#define MINDEX(d, i, j, m, n) ((d) * (m) * (n) + (i) * (n) + (j))

// l: large dimension s: small dimension
// e.g. ls indicates an m x n matrix where m is large and n is small

#define M_TILE 8
#define K_TILE 32
#define N_TILE 32
#define K_SPMM_TILE 32
#define K_TILE_SM (K_SPMM_TILE)
#define N_TILE_SM (N_TILE)
#define M_TILE_BLOCK (MM_CALC_PER_BLOCK * M_TILE)

// balance strategy for spmm_mm fused kernel
void kg_spmm_mm_balance(int m, int nnz, int *RowPtr, warp_info **winfo, int *winfo_n)
{
    std::vector<warp_info> host_winfo;

    // M_TILE partition
    for (int row = 0; row < m; row += M_TILE_BLOCK)
    {
        int row_st = row;
        int row_ed = kg_min(row + M_TILE_BLOCK, m);

        int idx_st = RowPtr[row_st];
        int idx_ed = RowPtr[row_ed];

        int nnz_per_warp = (idx_ed - idx_st + SPMM_PER_BLOCK - 1) / SPMM_PER_BLOCK;

        // printf("block: %d %d %d %d\n", row_st, row_ed, idx_st, idx_ed);
        // printf("nnz_per_warp %d\n", nnz_per_warp);

        warp_info tmp_info[SPMM_PER_BLOCK];
        int j = idx_st;
        for (int i = 0; i < SPMM_PER_BLOCK; i++)
        {
            int j_ed = kg_min(j + nnz_per_warp, idx_ed);
            tmp_info[i] = warp_info(-1, -1, j, j_ed);
            j = j_ed;
        }

        int current_row_st = row_st;
        int current_row_ed = row_st;
        for (int i = 0; i < SPMM_PER_BLOCK; i++)
        {
            while (RowPtr[current_row_st] < tmp_info[i].col_st)
                current_row_st++;
            while (RowPtr[current_row_ed] < tmp_info[i].col_ed)
                current_row_ed++;
            if (i > 0 && RowPtr[current_row_st] > tmp_info[i].col_st)
                tmp_info[i].row_st = current_row_st - 1;
            else
                tmp_info[i].row_st = current_row_st;
            tmp_info[i].row_ed = current_row_ed;

            host_winfo.push_back(tmp_info[i]);
        }

        // int i = row_st;
        // int current_row_st = i;
        // int j = idx_st;
        // while (i <= row_ed)
        // {
        //     while (j < idx_ed && j + nnz_per_warp <= RowPtr[i])
        //     {
        //         int j_ed = kg_min(j + nnz_per_warp, idx_ed);
        //         warp_info tmp = warp_info(current_row_st, i, j, j_ed);
        //         // printf("warp_info %d %d %d %d\n", current_row_st, i, j, j_ed);
        //         if (j_ed == RowPtr[i])
        //             current_row_st = i;
        //         else
        //             current_row_st = i - 1;
        //         j = j_ed;
        //         host_winfo.push_back(tmp);
        //     }
        //     i++;
        // }
    }

    *winfo_n = host_winfo.size();
    // for (int i = 0; i < *winfo_n; i++)
    // {
    //     warp_info info = host_winfo[i];
    //     if (i < 32)
    //         printf("warp %d: %d %d %d %d\n", i, info.row_st, info.row_ed, info.col_st, info.col_ed);
    // }

    cudaMalloc(winfo, host_winfo.size() * sizeof(warp_info));
    cudaMemcpy(*winfo, &host_winfo[0], host_winfo.size() * sizeof(warp_info), cudaMemcpyHostToDevice);
}

__global__ void kg_spmm_ls_ss_mm_pipeline_balanced(int m, int n, int k, int *RowPtr, int *ColIdx, 
float *input_a, float *input_b, float *output_c, warp_info *winfo, int winfo_n)
{
    extern __shared__ float s[];
    float *a_tile = (float*)&s[0];
    float *b_tile = (float*)&s[(MM_CALC_PER_BLOCK * M_TILE * K_SPMM_TILE) * 2];
    float *c_tile = (float*)&s[(MM_CALC_PER_BLOCK * M_TILE * K_SPMM_TILE + K_TILE * N_TILE_SM) * 2];

    // cg::thread_block tb = cg::this_thread_block();
    if (threadIdx.x < SPMM_PER_BLOCK * WARP_SIZE)
    {
        int block_row_st = winfo[blockIdx.x * SPMM_PER_BLOCK].row_st;
        // int w_ed = (blockIdx.x + 1) * SPMM_PER_BLOCK;
        // if (w_ed > winfo_n) w_ed = winfo_n;
        // int block_row_ed = winfo[w_ed].row_ed;

        int local_tid = threadIdx.x;
        int local_wid = local_tid / WARP_SIZE;
        int lane_id = local_tid & (WARP_SIZE - 1);

        int global_tid = blockIdx.x * SPMM_PER_BLOCK * WARP_SIZE + local_tid;
        int global_wid = global_tid / WARP_SIZE;

        int count = 0;

        for (int k_block = 0; k_block < k; k_block += K_SPMM_TILE)
        {
            int dbuff_i = count % 2;

            #pragma unroll
            for (int row = local_wid; row < M_TILE_BLOCK; row+=SPMM_PER_BLOCK)
            {
                #pragma unroll
                for (int col = lane_id; col < K_SPMM_TILE; col+=WARP_SIZE)
                    a_tile[MINDEX(dbuff_i, row, col, M_TILE_BLOCK, K_TILE_SM)] = 0;
            }

            asm volatile("bar.sync %0, %1;" : : "r"(3), "r"(SPMM_PER_BLOCK * WARP_SIZE) : "memory");

            int row_st = winfo[global_wid].row_st;
            int row_ed = winfo[global_wid].row_ed;
            int idx_st = winfo[global_wid].col_st;
            int idx_ed = winfo[global_wid].col_ed;

            for (int row = row_st; row < row_ed; row++)
            {
                int start_ptr = (RowPtr[row] > idx_st)? RowPtr[row]: idx_st;
                int end_ptr = (RowPtr[row + 1] < idx_ed)? RowPtr[row + 1]: idx_ed;
                float degree_inv = 1.0;
                // int self_idx = row * feat_len;

                #pragma unroll
                for (int j = lane_id; j < K_SPMM_TILE; j+=WARP_SIZE)
                {
                    float result = 0.0;
                    for (int i = start_ptr; i < end_ptr; i++)
                    {
                        int nid = ColIdx[i];
                        int feat_idx = nid * k + k_block + j;
                        result += input_a[feat_idx] * degree_inv;
                    }

                    // if (lane_id == 0 && local_wid == 15) printf("row %d result %.4f\n", row - block_row_st, result);
                    
                    atomicAdd(&a_tile[MINDEX(dbuff_i, row - block_row_st, j, M_TILE_BLOCK, K_SPMM_TILE)], result);
                    // if (row - block_row_st == 16 && (lane_id == 16 || lane_id == 0))
                    //     printf("result %.4f\n", a_tile[MINDEX(dbuff_i, row - block_row_st, j, M_TILE_BLOCK, K_SPMM_TILE)]);
                    //a_tile[MINDEX(dbuff_i, row - block_row_st, j, M_TILE_BLOCK, K_TILE)] = result;

                    // if (row < 5 && lane_id == 0)
                    //     printf("warp %d row %d result %.4f\n", global_wid, row, a_tile[MINDEX(dbuff_i, row - block_row_st, j, M_TILE_BLOCK, K_TILE)]);

                    // if (lane_id == 0 && local_wid == 0) 
                    //     printf("row %d result %.4f\n", row - block_row_st, a_tile[MINDEX(dbuff_i, row - block_row_st, j, M_TILE_BLOCK, K_TILE)]);
                }
                // atomicAdd(&out_feat[self_idx + j], result);
            }

            count++;

            asm volatile("bar.sync %0, %1;" : : "r"(0), 
            "r"((SPMM_PER_BLOCK + MM_CALC_PER_BLOCK + MM_MEM_PER_BLOCK) * WARP_SIZE) : "memory");
            //if (local_wid == 0 && lane_id == 0) printf("WG1 warp %d done global!\n", global_wid);
        }

    }
    else
    {
        int row_st = winfo[blockIdx.x * SPMM_PER_BLOCK].row_st;
        int w_ed = (blockIdx.x + 1) * SPMM_PER_BLOCK - 1;
        if (w_ed >= winfo_n) w_ed = winfo_n - 1;
        int row_ed = winfo[w_ed].row_ed;

        // need to ensure that (row_ed - row_st) <= M_TILE
        int m_len = kg_min(row_ed - row_st, M_TILE_BLOCK);

        int local_tid = (threadIdx.x - SPMM_PER_BLOCK * WARP_SIZE) % (MM_MEM_PER_BLOCK * WARP_SIZE);
        int local_wid = local_tid / WARP_SIZE;
        int lane_id = local_tid & (WARP_SIZE - 1);

        int global_tid = blockIdx.x * MM_MEM_PER_BLOCK * WARP_SIZE + local_tid;
        int global_wid = global_tid / WARP_SIZE;

        // if (lane_id == 0) printf("tid %d wid %d\n", global_tid, global_wid);

        #define MT_TILE (M_TILE / 4)
        #define NT_TILE (N_TILE / 8)

        int nt_tiles = N_TILE / NT_TILE;

        int nt_tile_idx = (local_tid % nt_tiles) * NT_TILE;
        int mt_tile_idx = (local_tid / nt_tiles) * MT_TILE;

        int input_a_st = global_wid * M_TILE;
        int m_local_st = local_wid * M_TILE;

        // if (lane_id == 0) printf("row_st, row_ed %d %d\n", row_st, row_ed);

        if (threadIdx.x < (MM_MEM_PER_BLOCK + SPMM_PER_BLOCK) * WARP_SIZE)
        {
            int count = 0;
            int count3 = 0;
            bool zfill = false;
            if (lane_id % 4 == 0) zfill = true;
            
            // K tile iteration
            for (int k_block = 0; k_block < k; k_block += K_SPMM_TILE)
            {
                for (int n_block = 0; n_block < n; n_block += N_TILE)
                {

                    for (int kk = k_block; kk < k_block + K_SPMM_TILE; kk+= K_TILE)
                    {
                        int dbuff_i = count % 2;

                        #pragma unroll
                        for (int i = local_tid; i < K_TILE * N_TILE; i+= MM_MEM_PER_BLOCK * WARP_SIZE)
                        {
                            int tgt_i = i / N_TILE;
                            int tgt_j = i % N_TILE;
                            b_tile[MINDEX(dbuff_i, 0, i, K_TILE, N_TILE_SM)] = 
                            input_b[(kk + tgt_i) * n + n_block + tgt_j];
                        }

                        count++;

                        if (n > N_TILE)
                        {
                            // read in C tile
                            if (k_block && kk == k_block)
                            {
                                #pragma unroll
                                for (int i = local_tid; i < M_TILE_BLOCK * N_TILE; i+= MM_MEM_PER_BLOCK * WARP_SIZE)
                                {
                                    int tgt_i = i / M_TILE_BLOCK;
                                    int tgt_j = i % M_TILE_BLOCK;
                                    if (tgt_i < m_len)
                                    {
                                        c_tile[MINDEX(dbuff_i, 0, i, M_TILE_BLOCK, N_TILE_SM)] =
                                        output_c[(row_st + tgt_i) * n + n_block + tgt_j];
                                    }                                
                                }
                            }

                            if (n_block == 0 && kk == k_block)
                            {
                                asm volatile("bar.sync %0, %1;" : : "r"(0), 
                                "r"((SPMM_PER_BLOCK + MM_CALC_PER_BLOCK + MM_MEM_PER_BLOCK) * WARP_SIZE) : "memory");
                                //if (lane_id == 0) printf("WG2 warp %d done global!\n", global_wid);
                            }
                            else
                            {
                                //if (lane_id == 0) printf("WG2 warp %d before local sync!\n", global_wid);
                                asm volatile("bar.sync %0, %1;" : : "r"(2), 
                                "r"((MM_CALC_PER_BLOCK + MM_MEM_PER_BLOCK) * WARP_SIZE) : "memory");
                                //if (lane_id == 0) printf("WG2 warp %d done local!\n", global_wid);
                            }
                        }
                        else
                        {
                            if (kk == k_block)
                            {
                                asm volatile("bar.sync %0, %1;" : : "r"(0), 
                                "r"((SPMM_PER_BLOCK + MM_CALC_PER_BLOCK + MM_MEM_PER_BLOCK) * WARP_SIZE) : "memory");
                                //if (local_wid == 0 && lane_id == 0) printf("WG2 warp %d done global!\n", global_wid);
                            }
                            else
                            {
                                asm volatile("bar.sync %0, %1;" : : "r"(2), 
                                "r"((MM_CALC_PER_BLOCK + MM_MEM_PER_BLOCK) * WARP_SIZE) : "memory");
                                //if (local_wid == 0 && lane_id == 0) printf("WG2 warp %d done local! %d\n", global_wid, kk);
                            }
                        }
                    }

                }
            }
        }
        else if (threadIdx.x < (SPMM_PER_BLOCK + MM_MEM_PER_BLOCK + MM_CALC_PER_BLOCK) * WARP_SIZE)
        {
            int count = 0;
            int count2 = 0;
            //if (local_tid == 0 && lane_id == 0) printf("threadIdx: %d\n", threadIdx.x);

            float c_thread_tile[MT_TILE][NT_TILE];

            for (int k_block = 0; k_block < k; k_block += K_SPMM_TILE)
            {
                for (int n_block = 0; n_block < n; n_block += N_TILE)
                {
                    int dbuff_i = count % 2;

                    for (int kk = k_block; kk < k_block + K_SPMM_TILE; kk+= K_TILE)
                    {
                        int dbuff_i2 = count2 % 2;

                        if (n > N_TILE)
                        {
                            if (n_block == 0 && kk == k_block)
                            {
                                asm volatile("bar.sync %0, %1;" : : "r"(0), 
                                "r"((SPMM_PER_BLOCK + MM_CALC_PER_BLOCK + MM_MEM_PER_BLOCK) * WARP_SIZE) : "memory");
                            }
                            else
                            {
                                //if (lane_id == 0) printf("WG3 warp %d before local sync!\n", global_wid);

                                asm volatile("bar.sync %0, %1;" : : "r"(2), 
                                "r"((MM_CALC_PER_BLOCK + MM_MEM_PER_BLOCK) * WARP_SIZE) : "memory");
                                // for (int i = 0; i < m_len; i++)
                                //     for (int j = 0; j < NT_TILE; j++)
                                //         c_thread_tile[i][j] = 
                                //         c_tile[MINDEX(dbuff_i, mt_tile_idx + i, nt_tile_idx + j, M_TILE_BLOCK, N_TILE_SM)];
                                
                                //if (lane_id == 0) printf("WG3 warp local %d done!\n", global_wid);
                            }
                            if (k_block == 0 && kk == k_block)
                            {
                                for (int i = 0; i < MT_TILE; i++)
                                    for (int j = 0; j < NT_TILE; j++)
                                        c_thread_tile[i][j] = 0;
                            }
                            else
                            {
                                for (int i = 0; i < MT_TILE; i++)
                                    for (int j = 0; j < NT_TILE; j++)
                                        //c_tile[(mt_tile_idx + i) * N_TILE + nt_tile_idx + j] = c_thread_tile[i][j];
                                        //if (mt_tile_idx + i < m_len)
                                        c_thread_tile[i][j] = c_tile[MINDEX(dbuff_i2, mt_tile_idx + i, nt_tile_idx + j, M_TILE_BLOCK, N_TILE_SM)];
                                // for (int i = 0; i < MT_TILE; i++)
                                //     for (int j = 0; j < NT_TILE; j++)
                                //         c_thread_tile[i][j] = 0;
                            }
                        }
                        else
                        {
                            if (kk == k_block)
                            {
                                asm volatile("bar.sync %0, %1;" : : "r"(0), 
                                "r"((SPMM_PER_BLOCK + MM_CALC_PER_BLOCK + MM_MEM_PER_BLOCK) * WARP_SIZE) : "memory");
                                //if (local_wid == 0 && lane_id == 0) printf("WG3 warp %d done global!\n", global_wid);
                            }
                            else
                            {
                                asm volatile("bar.sync %0, %1;" : : "r"(2), 
                                "r"((MM_CALC_PER_BLOCK + MM_MEM_PER_BLOCK) * WARP_SIZE) : "memory");
                                //if (local_wid == 0 && lane_id == 0) printf("WG3 warp %d done local! %d \n", global_wid, kk);
                            }

                            if (k_block == 0 && kk == k_block)
                            {
                                for (int i = 0; i < MT_TILE; i++)
                                    for (int j = 0; j < NT_TILE; j++)
                                        c_thread_tile[i][j] = 0;
                            }
                        }

                        float a_thread_tile[MT_TILE];
                        float b_thread_tile[NT_TILE];

                        // if (local_wid == 0 && lane_id == 0)
                        // {
                        //     for (int debugi = 0; debugi < MT_TILE; debugi++)
                        //         printf("%.4f\n", a_tile[MINDEX(dbuff_i, mt_tile_idx + debugi, 0, M_TILE_BLOCK, K_TILE_SM)]);
                        //     for (int debugi = 0; debugi < NT_TILE; debugi++)
                        //         printf("%.4f\n", b_tile[MINDEX(dbuff_i, 0, nt_tile_idx + debugi, K_TILE, N_TILE_SM)]);
                        // }

                        // if (local_wid == 0 && lane_id == 0) printf("tid %d %.4f %.4f %.4f\n", local_tid,
                        // a_thread_tile[0], b_thread_tile[0], c_thread_tile[0][0]);

                        #pragma unroll
                        for (int k = 0; k < K_TILE; k++)
                        {
                            #pragma unroll
                            for (int i = 0; i < MT_TILE; i++)
                            {
                                a_thread_tile[i] = a_tile[MINDEX(dbuff_i, mt_tile_idx + i, kk - k_block + k, M_TILE_BLOCK, K_SPMM_TILE)];
                                //if (local_wid == 0 && lane_id == 0) printf("a_thread_tile %.4f\n", a_thread_tile[i]);
                            }

                            #pragma unroll
                            for (int j = 0; j < NT_TILE; j++)
                            {
                                b_thread_tile[j] = b_tile[MINDEX(dbuff_i2, k, nt_tile_idx + j, K_TILE, N_TILE_SM)];
                                //if (local_wid == 0 && lane_id == 0) printf("b_thread_tile %.4f\n", b_thread_tile[j]);
                            }

                            // if (local_wid == 0 && lane_id == 0 && k == 0) printf("before tid %d %.4f %.4f %.4f\n", local_tid,
                            // a_thread_tile[0], b_thread_tile[0], c_thread_tile[0][0]);

                            #pragma unroll
                            for (int i = 0; i < MT_TILE; i++)
                                #pragma unroll
                                for (int j = 0; j < NT_TILE; j++)
                                {
                                    //c_thread_tile[i][j] += a_thread_tile[i] * b_thread_tile[j];
                                    c_thread_tile[i][j] = fmaf(a_thread_tile[i], b_thread_tile[j], c_thread_tile[i][j]);
                                    //if (local_wid == 0 && lane_id == 0) printf("%d %d c_thread_tile %.4f\n", i, j, c_thread_tile[i][j]);
                                }
                        }

                        // if (local_wid == 0 && lane_id == 0) printf("after tid %d %.4f %.4f %.4f\n", local_tid,
                        // a_thread_tile[0], b_thread_tile[0], c_thread_tile[0][0]);

                        //if (lane_id == 0) printf("warp here %d\n", local_wid);

                        // if (global_wid < 5 && lane_id == 0) printf("tid %d %.4f %.4f %.4f\n", local_tid,
                        // a_thread_tile[0], b_thread_tile[0], c_thread_tile[0][0]);

                        for (int i = 0; i < MT_TILE; i++)
                            for (int j = 0; j < NT_TILE; j++)
                                //c_tile[(mt_tile_idx + i) * N_TILE + nt_tile_idx + j] = c_thread_tile[i][j];
                                c_tile[MINDEX(dbuff_i2, mt_tile_idx + i, nt_tile_idx + j, M_TILE_BLOCK, N_TILE_SM)] = c_thread_tile[i][j];
                        
                        //if (local_tid == 0) printf("row_st m_len %d %d\n", row_st, m_len);

                        // write back C tile
                        for (int i = 0; i < M_TILE; i++)
                            if (m_local_st + i < m_len)
                                for (int j = lane_id; j < N_TILE; j+=WARP_SIZE)
                                    output_c[(row_st + m_local_st + i) * n + n_block + j] = c_tile[MINDEX(dbuff_i2, m_local_st + i, j, M_TILE_BLOCK, N_TILE_SM)];

                        // if (local_wid == 0 && lane_id == 0 && n_block == 32) 
                        // printf("write back %.4f\n", c_tile[MINDEX(dbuff_i2, mt_tile_idx, nt_tile_idx, M_TILE_BLOCK, N_TILE_SM)]);

                        // for (int i = 0; i < M_TILE; i++)
                        //     if (m_local_st + i < m_len)
                        //         for (int j = lane_id; j < K_TILE; j+=WARP_SIZE)
                        //             a_tile[MINDEX(dbuff_i, m_local_st + i, j, M_TILE_BLOCK, N_TILE_SM)] = 0;

                        // //write back C tile
                        // #pragma unroll
                        // for (int i = 0; i < M_TILE; i++)
                        // {
                        //     int tgt_i = i;
                        //     if (i < m_len)
                        //         #pragma unroll
                        //         for (int j = 0; j < N_TILE / WARP_SIZE; j++)
                        //         {
                        //             int tgt_j = j * WARP_SIZE + lane_id;
                        //             output_c[(row_st + tgt_i) * n + n_block + tgt_j] = 
                        //             c_tile[MINDEX(dbuff_i, tgt_i, tgt_j, M_TILE_BLOCK, N_TILE_SM)];
                        //         }
                        // }

                        count2++;
                    }
                    
                }

                count++;
            }
        }
    }
}

#undef MT_TILE
#undef NT_TILE
#undef KT_TILE

void kg_spmm_mm_pipeline_balance_execute(int m, int n, int k, int *rowptr, int *colidx, 
float *input_a, float *input_b, float *output_c, warp_info *winfo, int winfo_n)
{
    dim3 thread_num((SPMM_PER_BLOCK + MM_MEM_PER_BLOCK + MM_CALC_PER_BLOCK) * WARP_SIZE);
    //printf("%d\n", winfo_n);
    // printf("%d", thread_num);
    int mtile_pb = M_TILE * MM_CALC_PER_BLOCK;
    //dim3 block_num((m + mtile_pb - 1) / mtile_pb);
    dim3 block_num((winfo_n + SPMM_PER_BLOCK - 1) / SPMM_PER_BLOCK);

    int shared_size = 2 * (MM_CALC_PER_BLOCK * M_TILE * K_SPMM_TILE 
    + K_TILE * N_TILE_SM + MM_CALC_PER_BLOCK * M_TILE * N_TILE_SM) * sizeof(float);

    //printf("%d\n", shared_size);

    //printf("mtile_pb %d block_num %d\n", mtile_pb, (m + mtile_pb - 1) / mtile_pb);

    cudaFuncSetAttribute(kg_spmm_ls_ss_mm_pipeline_balanced, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size);

    kg_spmm_ls_ss_mm_pipeline_balanced<<<block_num, thread_num, shared_size>>>(m, n, k, rowptr, colidx, 
    input_a, input_b, output_c, winfo, winfo_n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

#undef M_TILE
#undef N_TILE
#undef K_TILE
