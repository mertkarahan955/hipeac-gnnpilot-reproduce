#include "../include/KG_GNN.h"
#include <cuda.h>
#include <stdio.h>
#include "cublas_v2.h"

#define MM_WARP_PER_BLOCK 4

#define MINDEX(d, i, j, m, n) ((d) * (m) * (n) + (i) * (n) + (j))

// l: large dimension s: small dimension
// e.g. ls indicates an m x n matrix where m is large and n is small

#define M_TILE 16
#define K_TILE 32
#define N_TILE 32
#define M_TILE_BLOCK (MM_WARP_PER_BLOCK * M_TILE)

__global__ void kg_ls_ss_mm_kernel(int m, int n, int k, 
float *input_a, float *input_b, float *output_c)
{
    int local_tid = threadIdx.x;
    int local_wid = local_tid / WARP_SIZE;
    int lane_id = local_tid & (WARP_SIZE - 1);

    // extern __shared__ float s[];
    // float *a_tile = (float*)&s[0];
    // float *b_tile = (float*)&s[(MM_WARP_PER_BLOCK * M_TILE * K_TILE) * 2];
    // float *c_tile = (float*)&s[(MM_WARP_PER_BLOCK * M_TILE * K_TILE + K_TILE * N_TILE) * 2];

#define MT_TILE 4
#define NT_TILE ((M_TILE * N_TILE) / WARP_SIZE / MT_TILE)
#define KT_TILE 8

    int nt_tiles = N_TILE / NT_TILE;

    int global_tid = blockIdx.x * blockDim.x + local_tid;
    int global_wid = global_tid / WARP_SIZE;

    int nt_tile_idx = (local_tid % nt_tiles) * NT_TILE;
    int mt_tile_idx = (local_tid / nt_tiles) * MT_TILE;

    int input_a_st = global_wid * M_TILE;
    int m_local_st = local_wid * M_TILE;

    if (input_a_st >= m) return;

    __shared__ float a_tile[2][MM_WARP_PER_BLOCK * M_TILE][K_TILE];
    __shared__ float b_tile[2][K_TILE][N_TILE];
    __shared__ float c_tile[MM_WARP_PER_BLOCK * M_TILE][N_TILE];

    int dbuff_i = 0;

    // N tile iteration
    for (int n_block = 0; n_block < n; n_block += N_TILE)
    {
        for (int i = 0; i < M_TILE; i++)
            for (int j = lane_id; j < N_TILE; j+=WARP_SIZE)
                c_tile[m_local_st + i][j] = 0;

        // read in A tile
        #pragma unroll
        for (int i = 0; i < M_TILE; i++)
            #pragma unroll
            for (int j = 0; j < K_TILE / WARP_SIZE; j++)
            {
                int tgt_j = j * WARP_SIZE + lane_id;
                a_tile[dbuff_i][m_local_st + i][tgt_j] = 
                input_a[input_a_st * k + i * k + 0 + tgt_j];
            }

        // read in B tile
        #pragma unroll
        for (int i = 0; i < K_TILE / MM_WARP_PER_BLOCK; i++)
        {
            int tgt_i = i + local_wid * K_TILE / MM_WARP_PER_BLOCK;
            #pragma unroll
            for (int j = 0; j < N_TILE / WARP_SIZE; j++)
            {
                int tgt_j = j * WARP_SIZE + lane_id;
                b_tile[dbuff_i][tgt_i][tgt_j] = 
                input_b[(0 + tgt_i) * n + n_block + tgt_j];
            }
        }

        // K tile iteration
        for (int k_block = 0; k_block < k; k_block += K_TILE)
        {

            __syncthreads();

            int dbuff_i_new = (dbuff_i + 1) % 2;
            //int dbuff_i_new = 0;

            float a_thread_tile[MT_TILE][KT_TILE];
            float b_thread_tile[KT_TILE][NT_TILE];
            float c_thread_tile[MT_TILE][NT_TILE];

            for (int i = 0; i < MT_TILE; i++)
                for (int j = 0; j < NT_TILE; j++)
                    c_thread_tile[i][j] = c_tile[mt_tile_idx + i][nt_tile_idx + j];

            for (int k = 0; k < K_TILE; k += KT_TILE)
            {
                for (int i = 0; i < MT_TILE; i++)
                    for (int kk = 0; kk < KT_TILE; kk++)
                        //a_thread_tile[i][kk] = a_tile[(mt_tile_idx + i) * K_TILE + k + kk];
                        a_thread_tile[i][kk] = a_tile[dbuff_i][mt_tile_idx + i][k + kk];
                for (int kk = 0; kk < KT_TILE; kk++)
                    for (int j = 0; j < NT_TILE; j++)
                        //b_thread_tile[kk][j] = b_tile[(k + kk) * N_TILE + nt_tile_idx + j];
                        b_thread_tile[kk][j] = b_tile[dbuff_i][k + kk][nt_tile_idx + j];

                for (int i = 0; i < MT_TILE; i++)
                    for (int j = 0; j < NT_TILE; j++)
                        for (int kk = 0; kk < KT_TILE; kk++)
                            c_thread_tile[i][j] += a_thread_tile[i][kk] * b_thread_tile[kk][j];
                        //c_thread_tile[i][j] += a_thread_tile[i][0] * b_thread_tile[0][j];
            }

            for (int i = 0; i < MT_TILE; i++)
                for (int j = 0; j < NT_TILE; j++)
                    //c_tile[(mt_tile_idx + i) * N_TILE + nt_tile_idx + j] = c_thread_tile[i][j];
                    c_tile[mt_tile_idx + i][nt_tile_idx + j] = c_thread_tile[i][j];

            int next_k_block = k_block + K_TILE;
            //if (next_k_block < k)
            {
                // read in A tile
                #pragma unroll
                for (int i = 0; i < M_TILE; i++)
                    //for (int j = lane_id; j < K_TILE; j+=WARP_SIZE)
                    #pragma unroll
                    for (int j = 0; j < K_TILE / WARP_SIZE; j++)
                    {
                        int tgt_j = j * WARP_SIZE + lane_id;
                        a_tile[dbuff_i_new][m_local_st + i][tgt_j] = 
                        input_a[input_a_st * k + i * k + next_k_block + tgt_j];
                    }

                // read in B tile
                #pragma unroll
                for (int i = 0; i < K_TILE / MM_WARP_PER_BLOCK; i++)
                {
                    int tgt_i = i + local_wid * K_TILE / MM_WARP_PER_BLOCK;
                    //for (int j = lane_id; j < N_TILE; j+=WARP_SIZE)
                    #pragma unroll
                    for (int j = 0; j < N_TILE / WARP_SIZE; j++)
                    {
                        int tgt_j = j * WARP_SIZE + lane_id;
                        b_tile[dbuff_i_new][tgt_i][tgt_j] = 
                        input_b[(next_k_block + tgt_i) * n + n_block + tgt_j];
                    }
                }
            }

            dbuff_i = dbuff_i_new;

            // for (int i = 0; i < MT_TILE; i++)
            // {
            //     for (int j = 0; j < NT_TILE; j++)
            //     {
            //         float res = 0.0;
            //         for (int k = 0; k < K_TILE; k++)
            //             res += a_tile[MINDEX(dbuff_i, mt_tile_idx + i, k, M_TILE_BLOCK, K_TILE)] 
            //                 * b_tile[MINDEX(dbuff_i, k, nt_tile_idx + j, K_TILE, N_TILE)];
            //         c_tile[MINDEX(0, mt_tile_idx + i, nt_tile_idx + j, M_TILE_BLOCK, N_TILE)] += res;
            //     }
            // }

            // for (int i = 0; i < MT_TILE; i++)
            // {
            //     for (int j = 0; j < NT_TILE; j++)
            //     {
            //         float res = 0.0;
            //         for (int k = 0; k < K_TILE; k++)
            //             res += a_tile[mt_tile_idx + i][k] * b_tile[k][nt_tile_idx + j];
            //         c_tile[mt_tile_idx + i][nt_tile_idx + j] += res;
            //     }
            // }
        }

        __syncthreads();
        
        // write back C tile
        for (int i = 0; i < M_TILE; i++)
            for (int j = lane_id; j < N_TILE; j+=WARP_SIZE)
                output_c[input_a_st * n + i * n + n_block + j] = c_tile[m_local_st + i][j];


        __syncthreads();
        
    }

    // __shared__ float a_tile[M_TILE][K_TILE];
    // __shared__ float b_tile[K_TILE][N_TILE];
    // __shared__ float c_tile[M_TILE][N_TILE];

    // // N tile iteration
    // for (int n_block = 0; n_block < n; n_block += N_TILE)
    // {
    //     // K tile iteration
    //     for (int k_block = 0; k_block < k; k_block += K_TILE)
    //     {

    //         // read in A tile
    //         for (int i = local_wid; i < M_TILE; i+=)
    //             for (int j = 0; j < K_TILE; j+=WARP_SIZE)
    //                 a_tile[m_local_st + i][j] = input_a[i * k + j];

    //         // read in B tile
    //         for (int i = 0; i < K_TILE; i++)
    //             for (int j = 0; j < N_TILE; j+=WARP_SIZE)
    //                 b_tile[i][j] = input_b[(k_block + i) * n + n_block + j];
            
    //         __syncthreads();

    //         for (int i = 0; i < M_TILE; i++)
    //         {
    //             for (int j = 0; j < N_TILE; j++)
    //             {
    //                 float res = 0.0;
    //                 for (int k = )
    //             }
    //         }

    //         __syncthreads();
            
    //         // write back C tile
    //         for (int i = 0; i < M_TILE; i++)
    //             for (int j = 0; j < N_TILE; j+=WARP_SIZE)
    //                 output_c[input_a_st * M_TILE * n + i * n + n_block + j] = c_tile[m_local_st + i][j];
    //     }
    // }

// #define MT_TILE 8
// #define NT_TILE 4
// #define KT_TILE 8
//             float a_thread_tile[MT_TILE][KT_TILE];
//             float b_thread_tile[KT_TILE][NT_TILE];
//             float c_thread_tile[MT_TILE][NT_TILE];

//             for (int i = 0; i < MT_TILE; i++)
//                 for (int j = 0; j < NT_TILE; j++)
//                 a_thread_tile[][]

// #undef MT_TILE
// #undef NT_TILE
// #undef KT_TILE

#undef MT_TILE
#undef NT_TILE
#undef KT_TILE

}

void kg_mm_execute(int m, int n, int k, float *input_a, float *input_b, float *output_c)
{
    dim3 thread_num(MM_WARP_PER_BLOCK * WARP_SIZE);
    int mtile_pb = M_TILE * MM_WARP_PER_BLOCK;
    dim3 block_num((m + mtile_pb - 1) / mtile_pb);

    int shared_size = (2 * (MM_WARP_PER_BLOCK * M_TILE * K_TILE 
    + K_TILE * N_TILE) + MM_WARP_PER_BLOCK * M_TILE * N_TILE) * sizeof(float);

    //printf("%d\n", shared_size);

    //printf("mtile_pb %d block_num %d\n", mtile_pb, (m + mtile_pb - 1) / mtile_pb);

    cudaFuncSetAttribute(kg_ls_ss_mm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size);

    kg_ls_ss_mm_kernel<<<block_num, thread_num>>>(m, n, k, input_a, input_b, output_c);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

#undef M_TILE
#undef N_TILE
#undef K_TILE

void cublas_mm_execute(int m, int n, int k, float *input_a, float *input_b, float *output_c)
{
    cublasStatus_t status;
    cublasHandle_t handle;
    status = cublasCreate(&handle);

    float alpha = 1.0, beta = 0.0;

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, 
    &alpha, input_a, k, input_b, n, &beta, output_c, m);
}