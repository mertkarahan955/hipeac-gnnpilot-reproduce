#include "../include/KG_GNN.h"
#include <cuda.h>
#include <stdio.h>
#include <cusparse.h>

#define WARP_ITER_SIZE 1

template <int FEAT_LEN>
__global__ void sddmm_aggregate_kernel_balance_aligned(int m, int nnz, int feat_len, int feat_st,
int *RowPtr, int *ColIdx, float *in_feat1, float *in_feat2, float *out_feat, warp_info* winfo, int winfo_n)
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

    // printf("active %d\n", lane_id);

    unsigned mask = __activemask();

    __shared__ float buffer_all[FEAT_LEN * WARP_PER_BLOCK];
    float *m1_buffer = buffer_all + FEAT_LEN * local_wid;

    for (int tgt = global_wid * WARP_ITER_SIZE; tgt < (global_wid + 1) * WARP_ITER_SIZE; tgt++)
    {
        if (tgt >= winfo_n) return;
        warp_info info = winfo[tgt];

        // if (!lane_id) printf("wid %d st %d ed %d\n", global_wid, info.row_st, info.row_ed);

        for (int row = info.row_st; row < info.row_ed; row++)
        {
            int start_ptr = (RowPtr[row] > info.col_st)? RowPtr[row]: info.col_st;
            int end_ptr = (RowPtr[row + 1] < info.col_ed)? RowPtr[row + 1]: info.col_ed;

            for (int j = j_st + lane_id; j < j_ed; j += WARP_SIZE)
            {
                m1_buffer[j - j_st] = in_feat1[row * feat_len + j];
            }

            for (int i = start_ptr; i < end_ptr; i++)
            {
                float result = 0.0;
                int nid = ColIdx[i];
                for (int j = j_st + lane_id; j < j_ed; j += WARP_SIZE)
                {
                    result += m1_buffer[j - j_st] * in_feat2[nid * feat_len + j];
                    // printf("tid %d row %d m1_buffer[0] = %.2f in_feat2 = %.2f\n", lane_id, row, m1_buffer[j - j_st], in_feat2[nid * feat_len + j]);
                }
                // if (result) printf("tid %d result %.2f\n", lane_id, result);
                for (int k = 16; k > 0; k >>= 1)
                {
                    result += __shfl_down_sync(mask, result, k);
                }
                if (lane_id == 0)
                {
                    //out_feat[i] = result;
                    atomicAdd(&out_feat[i], result);
                }
            }
        }
    }
}

void sddmm_aggregate_balance(int m, int nnz, int feat_len,
int *RowPtr, int *ColIdx, float *in_feat1, float *in_feat2, float *out_feat, warp_info* winfo, int winfo_n)
{
    // neighbour grouping for balance
    int warp_num = (winfo_n + WARP_ITER_SIZE - 1) / WARP_ITER_SIZE;
    int thread_num = warp_num * WARP_SIZE;
    int block_num = (thread_num + BLOCK_SIZE - 1) / BLOCK_SIZE;

    const int kernel_len = 32;
    dim3 grid(block_num, (feat_len + kernel_len - 1) / kernel_len);
    dim3 block(BLOCK_SIZE_ALIGN);
    int feat_st = 0;
    sddmm_aggregate_kernel_balance_aligned<kernel_len><<<grid, block>>>(m, nnz, feat_len, feat_st,
    RowPtr, ColIdx, in_feat1, in_feat2, out_feat, winfo, winfo_n);
}

float sddmm_aggregate_cusparse(int m, int nnz, int feat_len,
int *RowPtr, int *ColIdx, float *Values, float *in_feat1, float *in_feat2, int warmup, int repetitions)
{
    cusparseHandle_t handle;
    cusparseSpMatDescr_t csrDescr;
    cusparseDnMatDescr_t dnMatInputDescr1, dnMatInputDescr2;

    float alpha = 1.0f, beta = 0.0f;

    //printf("cuSPARSE version: %.d\n", cusparseGetVersion());

    CUSPARSE_CHECK(cusparseCreate(&handle));

    // creating sparse csr matrix
    CUSPARSE_CHECK(cusparseCreateCsr(&csrDescr, 
        m, m, nnz, RowPtr, ColIdx, Values, 
        CUSPARSE_INDEX_32I, // index 32-integer for indptr
        CUSPARSE_INDEX_32I, // index 32-integer for indices
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F          // datatype: 32-bit float real number
    ));

    // creating dense matrices 1
    CUSPARSE_CHECK(cusparseCreateDnMat(&dnMatInputDescr1,
                                        m,
                                        feat_len,
                                        m,
                                        in_feat1,
                                        CUDA_R_32F,
                                        CUSPARSE_ORDER_COL
    ));

    // creating dense matrices 2
    CUSPARSE_CHECK(cusparseCreateDnMat(&dnMatInputDescr2,
                                        feat_len,
                                        m,
                                        feat_len,
                                        in_feat2,
                                        CUDA_R_32F,
                                        CUSPARSE_ORDER_COL
    ));

    // allocate workspace buffer
    size_t workspace_size;
    CUSPARSE_CHECK(cusparseConstrainedGeMM_bufferSize(handle, 
    CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha, dnMatInputDescr1, dnMatInputDescr2,
    &beta, csrDescr, CUDA_R_32F, &workspace_size));

    void *workspace = NULL;
    CUDA_CHECK_ERROR(cudaMalloc(&workspace, workspace_size));

    // CUSPARSE_CHECK(cusparseSDDMM_preprocess(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
    // &alpha, dnMatInputDescr1, dnMatInputDescr2,
    // &beta, csrDescr, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, workspace));

    GpuTimer gpu_timer;
    // int warmup_iter = 10;
    // int repeat_iter = 100;
    for (int iter = 0; iter < warmup + repetitions; iter++) {
        if (iter == warmup) {
            cudaDeviceSynchronize();
            gpu_timer.start();
        }

        // run SpMM
        CUSPARSE_CHECK(cusparseConstrainedGeMM(handle, 
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, dnMatInputDescr1, dnMatInputDescr2,
        &beta, csrDescr, CUDA_R_32F, workspace));

        //cudaDeviceSynchronize();

    }
    gpu_timer.stop();

    float kernel_dur_usecs = gpu_timer.elapsed_msecs() * 1000 / repetitions;

    CUDA_CHECK_ERROR(cudaFree(workspace));

    return kernel_dur_usecs;

}