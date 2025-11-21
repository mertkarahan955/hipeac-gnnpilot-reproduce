#include "../include/KG_GNN.h"
#include <cuda.h>
#include <stdio.h>
#include "cublas_v2.h"
#include <cuda_pipeline.h>
// #include <cooperative_groups.h>
// #include <cooperative_groups/memcpy_async.h>

// namespace cg = cooperative_groups;

#define MM_PER_BLOCK 8
#define SPMM_PER_BLOCK 8

#define MINDEX(d, i, j, m, n) ((d) * (m) * (n) + (i) * (n) + (j))

// l: large dimension s: small dimension
// e.g. ls indicates an m x n matrix where m is large and n is small

#define M_TILE 32
#define K_TILE 8
#define N_TILE 32
#define K_SPMM_TILE 32
#define K_TILE_SM (K_TILE)
#define N_TILE_SM (N_TILE)
#define M_TILE_BLOCK (MM_PER_BLOCK * M_TILE)

// balance strategy for spmm_mm fused kernel
// M_TILE_BLOCK rows per block
void kg_spmm_mm_balance_final(int m, int nnz, int *RowPtr, warp_info **winfo, int *winfo_n)
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

__global__ void kg_spmm_ls_ss_mm_pipeline_final(int m, int n, int k, int *RowPtr, int *ColIdx, 
float *input_a, float *input_b, float *interm_c, float *output_c,
warp_info *winfo, int winfo_n, int m_start, int m_end)
{
    extern __shared__ float s[];
    float *a_tile = (float*)&s[0];
    float *b_tile = (float*)&s[(M_TILE_BLOCK * K_TILE) * 2];
    // float *c_tile = (float*)&s[(M_TILE_BLOCK * M_TILE * K_TILE + K_TILE * N_TILE_SM) * 2];

    if (m_start >= 0 && m_end > 0)
    {
        #define MT_TILE (M_TILE / 4)
        #define NT_TILE (N_TILE / 8)
        // #define MT_TILE 4
        // #define NT_TILE 4

        int local_tid = threadIdx.x;//- SPMM_PER_BLOCK * WARP_SIZE;
        int local_wid = local_tid / WARP_SIZE;
        int lane_id = local_tid & (WARP_SIZE - 1);

        // transposed mapping
        //int blocks = gridDim.x / (k / K_SPMM_TILE);
        int m_global_len = m_end - m_start;
        int m_global_tiles = (m_global_len + M_TILE_BLOCK - 1) / M_TILE_BLOCK;
        int n_global_tiles = n / N_TILE;

        int m_local_st = M_TILE * local_wid;

        // gridDim.x must be larger than m_global_tiles
        for (int global_bid = blockIdx.x; global_bid < m_global_tiles * n_global_tiles; global_bid += gridDim.x)
        {
            //if (local_tid == 0) printf("%d %d %d\n", global_bid, m_global_tiles * n_global_tiles, gridDim.x);

            // int bidx_x = global_bid % m_global_tiles;
            // int bidx_y = global_bid / m_global_tiles;
            int bidx_x = global_bid / n_global_tiles;
            int bidx_y = global_bid % n_global_tiles;

            int m_global_st = m_start + bidx_x * M_TILE_BLOCK;
            int n_global_st = bidx_y * N_TILE;

            // const int nt_tiles = 8;
            // const int mt_tiles = WARP_SIZE / nt_tiles;
            const int mt_tiles = M_TILE / MT_TILE;
            const int nt_tiles = N_TILE / NT_TILE;

            int mt_tile_st = lane_id / nt_tiles * MT_TILE;
            int nt_tile_st = lane_id % nt_tiles * NT_TILE;

            // if (lane_id == 0) 
            // printf("local_wid %d global_st %d %d tile_st %d %d\n", local_wid, m_global_st,
            // n_global_st, mt_tile_st, nt_tile_st);

            int count = 0;
            int dbuff_i = count % 2;

            float a_thread_tile[MT_TILE];
            float b_thread_tile[NT_TILE];
            float c_thread_tile[MT_TILE][NT_TILE];

            for (int i = 0; i < MT_TILE; i++)
                for (int j = 0; j < NT_TILE; j++)
                    c_thread_tile[i][j] = 0;

            // read in an (M_TILE * K_TILE) sized A tile
            #pragma unroll
            for (int i = 4 * lane_id; i < M_TILE * K_TILE; i+= 4 * WARP_SIZE)
            {
                int tgt_i = i / K_TILE;
                int tgt_j = i % K_TILE;

                // if (lane_id == 0 && local_wid == 0) printf("read i index: %d %.4f %.4f\n", m_global_st + m_local_st + tgt_i,
                // interm_c[(m_global_st + m_local_st + tgt_i) * k + tgt_j], interm_c[0]);

                if (m_global_st + m_local_st + tgt_i < m_end)
                {
                    size_t shm_ptr = __cvta_generic_to_shared(&a_tile[MINDEX(dbuff_i, m_local_st + tgt_i, tgt_j, M_TILE_BLOCK, K_TILE)]);
                    void* glb_ptr = (void*)&interm_c[(m_global_st + m_local_st + tgt_i) * k + tgt_j];
                    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" 
                    :: "l"(shm_ptr), "l"(glb_ptr), "n"(4 * sizeof(float)));                    
                }
                // {
                //     a_tile[MINDEX(dbuff_i, m_local_st + tgt_i, tgt_j, M_TILE_BLOCK, K_TILE)] = 
                //     interm_c[(m_global_st + m_local_st + tgt_i) * k + tgt_j];
                // }
                else
                    a_tile[MINDEX(dbuff_i, m_local_st + tgt_i, tgt_j, M_TILE_BLOCK, K_TILE)] = 0;
            }

            // collaboratively read in a (K_TILE * M_TILE) sized B tile
            #pragma unroll
            for (int i = 4 * local_tid; i < K_TILE * N_TILE; i+= 4 * MM_PER_BLOCK * WARP_SIZE)
            {
                int tgt_i = i / N_TILE;
                int tgt_j = i % N_TILE;
                size_t shm_ptr = __cvta_generic_to_shared(&b_tile[MINDEX(dbuff_i, tgt_i, tgt_j, K_TILE, N_TILE_SM)]);
                void* glb_ptr = (void*)&input_b[(tgt_i) * n + n_global_st + tgt_j];
                asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" 
                :: "l"(shm_ptr), "l"(glb_ptr), "n"(4 * sizeof(float)));
                // b_tile[MINDEX(dbuff_i, tgt_i, tgt_j, K_TILE, N_TILE_SM)] = 
                // input_b[(tgt_i) * n + n_global_st + tgt_j];
            }

            for (int k_block = 0; k_block < k; k_block += K_TILE)
            {
                dbuff_i = count % 2;
                int new_dbuff_i = (dbuff_i + 1) % 2;
                int next_k_block = k_block + K_TILE;

                asm volatile("cp.async.wait_all;\n" ::);

                asm volatile("bar.sync %0, %1;" : : "r"(1), 
                "r"((MM_PER_BLOCK) * WARP_SIZE) : "memory");

                if (next_k_block < k)
                {
                    // read in an (M_TILE * K_TILE) sized A tile
                    // #pragma unroll
                    // for (int i = 0; i < M_TILE; i++)
                    // {
                    //     int tgt_i = i;
                    //     #pragma unroll
                    //     for (int j = lane_id; j < K_TILE; j+=WARP_SIZE)
                    //     {
                    //         int tgt_j = j;
                    //         if (m_global_st + m_local_st + tgt_i < m_end)
                    //             a_tile[MINDEX(new_dbuff_i, m_local_st + tgt_i, tgt_j, M_TILE_BLOCK, K_TILE)] = 
                    //             interm_c[(m_global_st + m_local_st + tgt_i) * k + next_k_block + tgt_j];
                    //     }
                    // }

                    // read in an (M_TILE * K_TILE) sized A tile
                    #pragma unroll
                    for (int i = 4 * lane_id; i < M_TILE * K_TILE; i+= 4 * WARP_SIZE)
                    {
                        int tgt_i = i / K_TILE;
                        int tgt_j = i % K_TILE;
                        if (m_global_st + m_local_st + tgt_i < m_end)
                        // copy
                        // {
                        //     a_tile[MINDEX(new_dbuff_i, m_local_st + tgt_i, tgt_j, M_TILE_BLOCK, K_TILE)] = 
                        //     interm_c[(m_global_st + m_local_st + tgt_i) * k + next_k_block + tgt_j];
                        // }
                        // async copy
                        {
                            size_t shm_ptr = __cvta_generic_to_shared(&a_tile[MINDEX(new_dbuff_i, m_local_st + tgt_i, tgt_j, M_TILE_BLOCK, K_TILE)]);
                            void* glb_ptr = (void*)&interm_c[(m_global_st + m_local_st + tgt_i) * k + next_k_block + tgt_j];
                            asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" 
                            :: "l"(shm_ptr), "l"(glb_ptr), "n"(4 * sizeof(float)));
                        }
                        else
                            a_tile[MINDEX(dbuff_i, m_local_st + tgt_i, tgt_j, M_TILE_BLOCK, K_TILE)] = 0;
                    }

                    // collaboratively read in a (K_TILE * M_TILE) sized B tile
                    #pragma unroll
                    for (int i = 4 * local_tid; i < K_TILE * N_TILE; i+= 4 * MM_PER_BLOCK * WARP_SIZE)
                    {
                        int tgt_i = i / N_TILE;
                        int tgt_j = i % N_TILE;
                        // b_tile[MINDEX(new_dbuff_i, tgt_i, tgt_j, K_TILE, N_TILE_SM)] = 
                        // input_b[(next_k_block + tgt_i) * n + n_global_st + tgt_j];
                        size_t shm_ptr = __cvta_generic_to_shared(&b_tile[MINDEX(new_dbuff_i, tgt_i, tgt_j, K_TILE, N_TILE_SM)]);
                        void* glb_ptr = (void*)&input_b[(next_k_block + tgt_i) * n + n_global_st + tgt_j];
                        asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" 
                        :: "l"(shm_ptr), "l"(glb_ptr), "n"(4 * sizeof(float)));
                    }
                }

                #pragma unroll
                for (int k = 0; k < K_TILE; k++)
                {
                    #pragma unroll
                    for (int i = 0; i < MT_TILE; i++)
                    {
                        a_thread_tile[i] = a_tile[MINDEX(dbuff_i, m_local_st + mt_tile_st + i, k, M_TILE_BLOCK, K_TILE)];
                        //if (local_wid == 0 && lane_id == 0) printf("a_thread_tile %.4f\n", a_thread_tile[i]);
                    }

                    #pragma unroll
                    for (int j = 0; j < NT_TILE; j++)
                    {
                        b_thread_tile[j] = b_tile[MINDEX(dbuff_i, k, nt_tile_st + j, K_TILE, N_TILE_SM)];
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

                count++;

            }

            //if (local_wid == 0 && lane_id == 0) printf("%d %d\n", );

            //write back C tile
            for (int i = 0; i < MT_TILE; i++)
            {
                // if (lane_id == 0 && local_wid == 3)
                //     printf("???%d %d %d %d\n", m_global_st, m_local_st, mt_tile_st, m_end);

                if (m_global_st + m_local_st + mt_tile_st + i < m_end)
                    for (int j = 0; j < NT_TILE; j++)
                    {
                        // if (lane_id == 0 && local_wid == 0 && !i && !j)
                        //     printf("lane_id %d to %d %d info %d %d %d %d %.4f\n", lane_id, m_global_st + m_local_st + mt_tile_st + i,
                        //     n_global_st + nt_tile_st + j, m_global_st, m_local_st, mt_tile_st, nt_tile_st, c_thread_tile[0][0]);
                        output_c[(m_global_st + m_local_st + mt_tile_st + i) * n + n_global_st + nt_tile_st + j] =
                        c_thread_tile[i][j];
                    }
            }

        }

        #undef MT_TILE
        #undef NT_TILE

    }

    //if (threadIdx.x < SPMM_PER_BLOCK * WARP_SIZE)
    if (winfo_n > 0)
    {
        //int block_row_st = winfo[blockIdx.x * SPMM_PER_BLOCK].row_st;

        int local_tid = threadIdx.x;
        int local_wid = local_tid / WARP_SIZE;
        int lane_id = local_tid & (WARP_SIZE - 1);

        // transposed mapping
        int blocks = gridDim.x / (k / K_SPMM_TILE);
        int bidx_x = blockIdx.x % blocks;
        int bidx_y = blockIdx.x / blocks;

        // int bidx_x = blockIdx.x / (k / K_SPMM_TILE);
        // int bidx_y = blockIdx.x % (k / K_SPMM_TILE);

        int global_tid = bidx_x * SPMM_PER_BLOCK * WARP_SIZE + local_tid;
        int global_wid = global_tid / WARP_SIZE;

        int j = WARP_SIZE * bidx_y + lane_id;
        if (j >= k) return;

        int tgt = global_wid;
        if (tgt >= winfo_n) return;

        warp_info info = winfo[tgt];

        for (int row = info.row_st; row < info.row_ed; row++)
        {
            int start_ptr = (RowPtr[row] > info.col_st)? RowPtr[row]: info.col_st;
            int end_ptr = (RowPtr[row + 1] < info.col_ed)? RowPtr[row + 1]: info.col_ed;
            float degree_inv = 1.0 ;
            int self_idx = row * k;
            float result = 0.0;
            for (int i = start_ptr; i < end_ptr; i++)
            {
                int nid = ColIdx[i];
                int feat_idx = nid * k + j;
                result += input_a[feat_idx] * degree_inv;
            }
            
            atomicAdd(&interm_c[self_idx + j], result);
            
            // if (lane_id == 0)
            // printf("warp %d st %d ed %d %.4f %.4f\n", global_wid, info.row_st, info.row_ed, result, interm_c[self_idx + j]);
        }
    }
    //else
}

extern __global__ void gcn_aggregate_kernel_balance42(int m, int nnz, int feat_len, int feat_st, int feat_size,
int *RowPtr, int *ColIdx, float *in_feat, float *out_feat, warp_info* winfo, int winfo_n);

void kg_spmm_mm_pipeline_final(int m, int n, int k, int *rowptr, int *colidx, 
float *input_a, float *input_b, float *interm_c, float *output_c, warp_info *winfo, int winfo_n)
{
    //const int expected_block_per_sm = 2;
    //const int warps_per_wave = SM_NUM * expected_block_per_sm * SPMM_PER_BLOCK * 4;
    int waves = 4;
    const int warps_per_wave = kg_max((winfo_n + waves - 1) / waves, 1);
    //const int m_per_wave = warps_per_wave / SPMM_PER_BLOCK * M_TILE_BLOCK;
    const int m_per_wave = (m + waves - 1)/ waves;
    //printf("winfo_n %d %d\n", winfo_n, m_per_wave);

    int m_start = 0;
    int m_end = m_per_wave;

    int i_end = kg_min(winfo_n, warps_per_wave);
    int winfo_n_current = i_end;

    dim3 thread_num((SPMM_PER_BLOCK) * WARP_SIZE);
    dim3 block_num((winfo_n_current + SPMM_PER_BLOCK - 1) / SPMM_PER_BLOCK * k / K_SPMM_TILE);

    int shared_size = 2 * (SPMM_PER_BLOCK * M_TILE * K_TILE 
    + K_TILE * N_TILE_SM) * sizeof(float);

    cudaFuncSetAttribute(kg_spmm_ls_ss_mm_pipeline_final, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size);

    kg_spmm_ls_ss_mm_pipeline_final<<<block_num, thread_num, shared_size>>>(m, n, k, rowptr, colidx, 
    input_a, input_b, interm_c, output_c, winfo, winfo_n_current, -1, -1);
    cudaDeviceSynchronize();

    for (int i = warps_per_wave; i < winfo_n; i+=warps_per_wave)
    {
        i_end = kg_min(winfo_n, i + warps_per_wave);
        winfo_n_current = i_end - i;

        block_num = dim3((winfo_n_current + SPMM_PER_BLOCK - 1) / SPMM_PER_BLOCK * k / K_SPMM_TILE);

        // int shared_size = 2 * (MM_PER_BLOCK * M_TILE * K_TILE 
        // + K_TILE * N_TILE_SM + MM_PER_BLOCK * M_TILE * N_TILE_SM) * sizeof(float);

        //printf("%d %d\n", m_start, m_end);

        kg_spmm_ls_ss_mm_pipeline_final<<<block_num, thread_num, shared_size>>>(m, n, k, rowptr, colidx, 
        input_a, input_b, interm_c, output_c, winfo + i, winfo_n_current, m_start, m_end);

        cudaDeviceSynchronize();

        m_start = m_end;
        m_end = kg_min(m_end + m_per_wave, m);

        //printf("m_start m_end %d %d\n", m_start, m_end);

        // cudaError_t err = cudaGetLastError();
        // if (err != cudaSuccess) {
        //     printf("CUDA Error: %s\n", cudaGetErrorString(err));
        // }
        // int warp_num = winfo_n;
        // int thread_num = warp_num * WARP_SIZE;
        // int block_num = (thread_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
        // dim3 grid(block_num, k / WARP_SIZE);
        // dim3 block(BLOCK_SIZE_ALIGN);
        // int feat_st = 0;
        // gcn_aggregate_kernel_balance42<<<grid, block>>>(m, 0, k, feat_st, WARP_SIZE,
        // rowptr, colidx, input_a, interm_c, winfo + i, winfo_n_current);
    }

    // printf("warps %d\n", winfo_n_current);
    //printf("%d %d\n", m_start, m_end);

    kg_spmm_ls_ss_mm_pipeline_final<<<block_num, thread_num, shared_size>>>(m, n, k, rowptr, colidx, 
    input_a, input_b, interm_c, output_c, winfo, 0, m_start, m_end);

    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("CUDA Error: %s\n", cudaGetErrorString(err));
    // }

//     cudaDeviceSynchronize();
}

#undef M_TILE
#undef N_TILE
#undef K_TILE
