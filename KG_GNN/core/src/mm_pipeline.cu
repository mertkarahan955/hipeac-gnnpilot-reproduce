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

#define MINDEX(d, i, j, m, n) ((d) * (m) * (n) + (i) * (n) + (j))

// l: large dimension s: small dimension
// e.g. ls indicates an m x n matrix where m is large and n is small

#define M_TILE 32
#define K_TILE 32
#define N_TILE 32
#define K_TILE_SM (K_TILE)
#define N_TILE_SM (N_TILE)
#define M_TILE_BLOCK (MM_CALC_PER_BLOCK * M_TILE)

__global__ void kg_ls_ss_mm_pipeline(int m, int n, int k, float *input_a, float *input_b, float *output_c)
{
    int local_tid = threadIdx.x % (MM_MEM_PER_BLOCK * WARP_SIZE);
    int local_wid = local_tid / WARP_SIZE;
    int lane_id = local_tid & (WARP_SIZE - 1);

    int global_tid = blockIdx.x * MM_MEM_PER_BLOCK * WARP_SIZE + local_tid;
    int global_wid = global_tid / WARP_SIZE;

#define MT_TILE (M_TILE / 4)
#define NT_TILE (N_TILE / 8)
//#define NT_TILE ((M_TILE * N_TILE) / WARP_SIZE / MT_TILE)
#define KT_TILE 4

    int nt_tiles = N_TILE / NT_TILE;

    int nt_tile_idx = (local_tid % nt_tiles) * NT_TILE;
    int mt_tile_idx = (local_tid / nt_tiles) * MT_TILE;

    extern __shared__ float s[];
    float *a_tile = (float*)&s[0];
    float *b_tile = (float*)&s[(MM_CALC_PER_BLOCK * M_TILE * K_TILE_SM) * 2];
    float *c_tile = (float*)&s[(MM_CALC_PER_BLOCK * M_TILE * K_TILE_SM + K_TILE * N_TILE_SM) * 2];

    int input_a_st = global_wid * M_TILE;
    int m_local_st = local_wid * M_TILE;

    // cg::thread_block tb = cg::this_thread_block();

    if (threadIdx.x < MM_MEM_PER_BLOCK * WARP_SIZE)
    {
        //if (local_tid == 0 && lane_id == 0) printf("threadIdx: %d\n", threadIdx.x);

        int count = 0;
        // int ln_block = -1, lk_block = -1;
        // int lln_block = -1, llk_block = -1;

        bool zfill = false;

        if (lane_id % 4 == 0) zfill = true;
        int src_in_bytes = 0;

        for (int n_block = 0; n_block < n; n_block += N_TILE)
        {
            // K tile iteration
            for (int k_block = 0; k_block < k; k_block += K_TILE)
            {
                int dbuff_i = count % 2;

                int j_st = lane_id % 8;
                int i_st = lane_id / 8;

                // #pragma unroll
                // for (int i = i_st; i < M_TILE; i += 4)
                // {
                //     int tgt_i = i;
                //     #pragma unroll
                //     for (int j = j_st * 4; j < K_TILE; j += 32)
                //     {
                //         int tgt_j = j;
                //         size_t shm_ptr = __cvta_generic_to_shared(&a_tile[MINDEX(dbuff_i, m_local_st + i, tgt_j, M_TILE_BLOCK, K_TILE_SM)]);
                //         void* glb_ptr = (void*)&input_a[input_a_st * k + i * k + k_block + tgt_j];
                //         asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" 
                //         :: "l"(shm_ptr), "l"(glb_ptr), "n"(4 * sizeof(float)));
                //     }
                // }

                if (zfill)
                {
                    #pragma unroll
                    for (int i = 0; i < M_TILE; i++)
                    //for (int j = lane_id; j < K_TILE; j+=WARP_SIZE)
                    {
                        #pragma unroll
                        for (int j = 0; j < K_TILE / WARP_SIZE; j++)
                        {
                            int tgt_j = j * WARP_SIZE + lane_id;
                            // a_tile[MINDEX(dbuff_i, m_local_st + i, tgt_j, M_TILE_BLOCK, K_TILE)] = 
                            // input_a[input_a_st * k + i * k + k_block + tgt_j];
                            // cg::memcpy_async(tb, &a_tile[MINDEX(dbuff_i, m_local_st + i, tgt_j, M_TILE_BLOCK, K_TILE)], 
                            // &input_a[input_a_st * k + i * k + k_block + tgt_j], sizeof(float));
                            // __pipeline_memcpy_async(&a_tile[MINDEX(dbuff_i, m_local_st + i, tgt_j, M_TILE_BLOCK, K_TILE_SM)],
                            // &input_a[input_a_st * k + i * k + k_block + tgt_j], sizeof(float));
                            // __pipeline_memcpy_async(&a_tile[MINDEX(dbuff_i, m_local_st + i, tgt_j, M_TILE_BLOCK, K_TILE)],
                            // &input_a[input_a_st * k + i * k + k_block + tgt_j], sizeof(float));
                            // cg::memcpy_async(tb, local_smem, elementsInShared, global_data + index, elementsPerThreadBlock - index);
                            // copy_count = min(elementsInShared, elementsPerThreadBlock - index);
                            // cg::wait(tb);
                            //void* shm_ptr = (void*)&a_tile[MINDEX(dbuff_i, m_local_st + i, tgt_j, M_TILE_BLOCK, K_TILE_SM)];
                            size_t shm_ptr = __cvta_generic_to_shared(&a_tile[MINDEX(dbuff_i, m_local_st + i, tgt_j, M_TILE_BLOCK, K_TILE_SM)]);
                            void* glb_ptr = (void*)&input_a[input_a_st * k + i * k + k_block + tgt_j];
                            asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" 
                            :: "l"(shm_ptr), "l"(glb_ptr), "n"(4 * sizeof(float)));
                            // asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16;\n"
                            // ::"r"(shm_ptr), "l"(glb_ptr));
                        }
                    }
                }

                // #pragma unroll
                // for (int i = 0; i < M_TILE; i++)
                // //for (int j = lane_id; j < K_TILE; j+=WARP_SIZE)
                // {
                //     #pragma unroll
                //     for (int j = 0; j < K_TILE / WARP_SIZE; j++)
                //     {
                //         int tgt_j = j * WARP_SIZE + lane_id;
                //         a_tile[MINDEX(dbuff_i, m_local_st + i, tgt_j, M_TILE_BLOCK, K_TILE)] = 
                //         input_a[input_a_st * k + i * k + k_block + tgt_j];
                //     }
                // }

                // read in B tile
                #pragma unroll
                for (int i = 0; i < K_TILE / MM_MEM_PER_BLOCK; i++)
                {
                    int tgt_i = i + local_wid * K_TILE / MM_MEM_PER_BLOCK;
                    //for (int j = lane_id; j < N_TILE; j+=WARP_SIZE)
                    #pragma unroll
                    for (int j = 0; j < N_TILE / WARP_SIZE; j++)
                    {
                        int tgt_j = j * WARP_SIZE + lane_id;
                        b_tile[MINDEX(dbuff_i, tgt_i, tgt_j, K_TILE, N_TILE)] = 
                        input_b[(k_block + tgt_i) * n + n_block + tgt_j];
                        // cg::memcpy_async(tb, &b_tile[MINDEX(dbuff_i, tgt_i, tgt_j, K_TILE, N_TILE)], 
                        // &input_b[(k_block + tgt_i) * n + n_block + tgt_j], sizeof(float));
                        // // __pipeline_memcpy_async(&b_tile[MINDEX(dbuff_i, tgt_i, tgt_j, K_TILE, N_TILE_SM)],
                        // &input_b[(k_block + tgt_i) * n + n_block + tgt_j], sizeof(float));
                    }
                } 

                // __pipeline_commit();
                // __pipeline_wait_prior(0);

                asm volatile("cp.async.wait_all;\n" ::);

                // asm volatile("cp.async.commit_group;\n" ::);
                // asm volatile("cp.async.wait_group %0;\n" :: "n"(0));

                count++;

                //dbuff_i = (dbuff_i + 1) % 2;
                // llk_block = lk_block;
                // lln_block = ln_block;
                // lk_block = k_block;
                // ln_block = n_block;
                //asm volatile("bar.sync %0, %1;" : : "r"(local_wid), "r"(64) : "memory");
                __syncthreads();
            }

            // // write back last C tile
            // if (count >= 2)
            // {
            //     for (int i = 0; i < M_TILE; i++)
            //         for (int j = lane_id; j < N_TILE; j+=WARP_SIZE)
            //             output_c[input_a_st * n + i * n + n_block + j] = c_tile[MINDEX(dbuff_i, m_local_st + i, j, M_TILE_BLOCK, N_TILE)];
            // }
        }
    }
    else if (threadIdx.x < (MM_MEM_PER_BLOCK + MM_CALC_PER_BLOCK) * WARP_SIZE)
    {
        int dbuff_i = 0;

        //if (local_tid == 0 && lane_id == 0) printf("threadIdx: %d\n", threadIdx.x);

        for (int n_block = 0; n_block < n; n_block += N_TILE)
        {
            // float a_thread_tile[MT_TILE][KT_TILE];
            // float b_thread_tile[KT_TILE][NT_TILE];
            float c_thread_tile[MT_TILE][NT_TILE];

            for (int i = 0; i < MT_TILE; i++)
                for (int j = 0; j < NT_TILE; j++)
                    c_thread_tile[i][j] = 0;

            // K tile iteration
            #pragma unroll
            for (int k_block = 0; k_block < k; k_block += K_TILE)
            {
                //asm volatile("bar.sync %0, %1;" : : "r"(local_wid), "r"(64) : "memory");
                __syncthreads();

                // #pragma unroll
                // for (int k = 0; k < K_TILE; k += KT_TILE)
                // {
                //     for (int i = 0; i < MT_TILE; i++)
                //         for (int kk = 0; kk < KT_TILE; kk++)
                //             //a_thread_tile[i][kk] = a_tile[(mt_tile_idx + i) * K_TILE + k + kk];
                //             a_thread_tile[i][kk] = a_tile[MINDEX(dbuff_i, mt_tile_idx + i, k + kk, M_TILE_BLOCK, K_TILE_SM)];
                //     for (int kk = 0; kk < KT_TILE; kk++)
                //         for (int j = 0; j < NT_TILE; j++)
                //             //b_thread_tile[kk][j] = b_tile[(k + kk) * N_TILE + nt_tile_idx + j];
                //             b_thread_tile[kk][j] = b_tile[MINDEX(dbuff_i, k + kk, nt_tile_idx + j, K_TILE, N_TILE_SM)];

                //     for (int i = 0; i < MT_TILE; i++)
                //         for (int j = 0; j < NT_TILE; j++)
                //             for (int kk = 0; kk < KT_TILE; kk++)
                //                 c_thread_tile[i][j] += a_thread_tile[i][kk] * b_thread_tile[kk][j];
                //             //c_thread_tile[i][j] += a_thread_tile[i][0] * b_thread_tile[0][j];
                // }

                float a_thread_tile[MT_TILE];
                float b_thread_tile[NT_TILE];

                #pragma unroll
                for (int k = 0; k < K_TILE; k++)
                {
                    #pragma unroll
                    for (int i = 0; i < MT_TILE; i++)
                        a_thread_tile[i] = a_tile[MINDEX(dbuff_i, mt_tile_idx + i, k, M_TILE_BLOCK, K_TILE_SM)];

                    #pragma unroll
                    for (int j = 0; j < NT_TILE; j++)
                        b_thread_tile[j] = b_tile[MINDEX(dbuff_i, k, nt_tile_idx + j, K_TILE, N_TILE_SM)];

                    #pragma unroll
                    for (int i = 0; i < MT_TILE; i++)
                        #pragma unroll
                        for (int j = 0; j < NT_TILE; j++)
                            //c_thread_tile[i][j] += a_thread_tile[i] * b_thread_tile[j];
                            c_thread_tile[i][j] = fmaf(a_thread_tile[i], b_thread_tile[j], c_thread_tile[i][j]);

                }

                // if (local_tid == 0 && lane_id == 0)
                // {
                //     printf("a_thread_tile, %.4f\n", a_thread_tile[0]);
                //     printf("b_thread_tile, %.4f\n", b_thread_tile[0]);
                //     printf("c_thread_tile, %.4f\n", c_thread_tile[0][0]);
                // }

                dbuff_i = (dbuff_i + 1) % 2;

            }

            for (int i = 0; i < MT_TILE; i++)
                for (int j = 0; j < NT_TILE; j++)
                    //c_tile[(mt_tile_idx + i) * N_TILE + nt_tile_idx + j] = c_thread_tile[i][j];
                    c_tile[MINDEX(0, mt_tile_idx + i, nt_tile_idx + j, M_TILE_BLOCK, N_TILE_SM)] = c_thread_tile[i][j];

            // write back C tile
            for (int i = 0; i < M_TILE; i++)
                if (input_a_st + i < m)
                    for (int j = lane_id; j < N_TILE; j+=WARP_SIZE)
                        output_c[input_a_st * n + i * n + n_block + j] = c_tile[MINDEX(0, m_local_st + i, j, M_TILE_BLOCK, N_TILE_SM)];

        }
    }

}

#undef MT_TILE
#undef NT_TILE
#undef KT_TILE

void kg_mm_pipeline_execute(int m, int n, int k, float *input_a, float *input_b, float *output_c)
{
    dim3 thread_num((MM_MEM_PER_BLOCK + MM_CALC_PER_BLOCK) * WARP_SIZE);
    int mtile_pb = M_TILE * MM_CALC_PER_BLOCK;
    dim3 block_num((m + mtile_pb - 1) / mtile_pb);

    int shared_size = (2 * (MM_CALC_PER_BLOCK * M_TILE * K_TILE_SM 
    + K_TILE * N_TILE_SM) + MM_CALC_PER_BLOCK * M_TILE * N_TILE_SM) * sizeof(float);

    //printf("%d\n", shared_size);

    //printf("mtile_pb %d block_num %d\n", mtile_pb, (m + mtile_pb - 1) / mtile_pb);

    cudaFuncSetAttribute(kg_ls_ss_mm_pipeline, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size);

    kg_ls_ss_mm_pipeline<<<block_num, thread_num, shared_size>>>(m, n, k, input_a, input_b, output_c);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

#undef M_TILE
#undef N_TILE
#undef K_TILE