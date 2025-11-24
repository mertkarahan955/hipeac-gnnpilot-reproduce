#include <cuda.h>
#include <stdio.h>
#include <cusparse.h>
#include <sys/time.h>
#include "preprocessing.h"

__global__ void cuda_partition_block_sequential(int m, int nnz, int *RowPtr, int *ColIdx, int pck_size,
row_panel *binfo, int *binfo_n, row_panel *einfo, int *einfo_n, neighbor_group *nginfo, int *nginfo_n, bool long_dynamic)
{
    if (threadIdx.x == 0)
    {
        int large_nnz = 0;

        int current_size = 0;
        int last_start = 0;
        int i = 0;
        int block_num = 0;
        int eblock_num = 0;
        int ng_num = 0;
        while (i < m)
        {
            int row_len = RowPtr[i + 1] - RowPtr[i];
            for (int col_st = RowPtr[i]; col_st < RowPtr[i + 1]; col_st += NG_SIZE)
            {
                nginfo[ng_num++] = neighbor_group(i, col_st);
            }
            if (row_len > SINGLE_PCK_THRESH)
            {
                if (last_start < i)
                {
                    binfo[block_num++] = row_panel(last_start, i);
                    einfo[eblock_num++] = row_panel(last_start, i);
                    //printf("%d: %d %d\n", block_num, last_start, i);
                }
                // if (long_dynamic)
                for (int col_st = RowPtr[i]; col_st < RowPtr[i + 1]; col_st += LONG_BLOCK_NNZ)
                {
                    int col_ed = col_st + LONG_BLOCK_NNZ;
                    if (col_ed >= RowPtr[i + 1]) col_ed = RowPtr[i + 1];
                    einfo[eblock_num++] = row_panel(i, i + 1, col_st, col_ed);
                }
                //else
                binfo[block_num++] = row_panel(i, i + 1, 0, 0);

                large_nnz += row_len;
                //printf("%d: %d %d\n", block_num, i, i + 1);
                last_start = i + 1;
                current_size = 0;
            }
            else if (row_len + current_size >= pck_size)
            {
                binfo[block_num++] = row_panel(last_start, i + 1);
                einfo[eblock_num++] = row_panel(last_start, i + 1);
                //printf("%d: %d %d\n", block_num, last_start, i + 1);
                last_start = i + 1;
                current_size = 0;
            }
            else
            {
                current_size += row_len;
            }
            i++;
        }

        if (current_size)
        {
            binfo[block_num++] = row_panel(last_start, m);
            einfo[eblock_num++] = row_panel(last_start, m);
            //printf("%d: %d %d\n", block_num, last_start, m);
        }

        *binfo_n = block_num;
        *einfo_n = eblock_num;
        *nginfo_n = ng_num;

        printf("long row nnz: %d\n", large_nnz);
    }
}

void flash_partition(int m, int nnz, int *RowPtr, int *ColIdx, int pck_size,
row_panel *binfo, int *binfo_n, row_panel *einfo, int *einfo_n, neighbor_group *nginfo, int *nginfo_n, bool long_dynamic)
{
    cuda_partition_block_sequential<<<1, 32>>>(m, nnz, RowPtr, ColIdx, pck_size, binfo, binfo_n, 
    einfo, einfo_n, nginfo, nginfo_n, long_dynamic);
}

int64_t preprocessing_cuda(int m, int nnz, int *RowPtr, int *ColIdx, bool long_dynamic)
{
    kg_info* info = new kg_info();
    cudaMalloc(&info->rp_info, sizeof(row_panel) * m);
    cudaMalloc(&info->rp_n, sizeof(int));
    cudaMalloc(&info->ep_info, sizeof(row_panel) * nnz);
    cudaMalloc(&info->ep_n, sizeof(int));
    cudaMalloc(&info->ng_info, sizeof(neighbor_group) * nnz);
    cudaMalloc(&info->ng_n, sizeof(int));

    flash_partition(m, nnz, RowPtr, ColIdx, SINGLE_PCK_THRESH, info->rp_info, info->rp_n, 
    info->ep_info, info->ep_n, info->ng_info, info->ng_n, long_dynamic);

    cudaMemcpy(&(info->rp_n_host), info->rp_n, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&(info->ep_n_host), info->ep_n, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&(info->ng_n_host), info->ng_n, sizeof(int), cudaMemcpyDeviceToHost);

    return (int64_t)info;
}