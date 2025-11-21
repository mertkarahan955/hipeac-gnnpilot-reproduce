#include "../include/KG_GNN.h"
#include <cuda.h>
#include <stdio.h>
#include <cusparse.h>
#include <sys/time.h>

__global__ void cuda_partition_block_sequential(int m, int nnz, int *RowPtr, int *ColIdx, int pck_size,
block_info2 *binfo, int *binfo_n)
{
    if (threadIdx.x == 0)
    {
        int large_nnz = 0;

        int current_size = 0;
        int last_start = 0;
        int i = 0;
        int block_num = 0;
        while (i < m)
        {
            int row_len = RowPtr[i + 1] - RowPtr[i];
            if (row_len > SINGLE_PCK_THRESH)
            {
                if (last_start < i)
                {
                    binfo[block_num++] = block_info2(last_start, i);
                    //printf("%d: %d %d\n", block_num, last_start, i);
                }
                binfo[block_num++] = block_info2(i, i + 1);

                large_nnz += row_len;
                //printf("%d: %d %d\n", block_num, i, i + 1);
                last_start = i + 1;
                current_size = 0;
            }
            else if (row_len + current_size >= pck_size)
            {
                binfo[block_num++] = block_info2(last_start, i + 1);
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
            binfo[block_num++] = block_info2(last_start, m);
            //printf("%d: %d %d\n", block_num, last_start, m);
        }

        *binfo_n = block_num;

        printf("long row nnz: %d\n", large_nnz);
    }
}

void flash_partition_simple(int m, int nnz, int *RowPtr, int *ColIdx, int pck_size,
block_info2 *binfo, int *binfo_n)
{
    cuda_partition_block_sequential<<<1, 32>>>(m, nnz, RowPtr, ColIdx, pck_size, binfo, binfo_n);
}