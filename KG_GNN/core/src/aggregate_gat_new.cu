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

                    // if (lane_id == 0 && row == 0)
                    // {
                    //     printf("%d %.3f %.3f\n", nid, in_feat[feat_idx], edge_weight[i]);
                    // }
                }
                atomicAdd(&out_feat[self_idx + j], result / sum_vec[row]);
                // if (lane_id == 0)
                // {
                //     printf("row %d result %.3f\n", row, out_feat[self_idx + j]);
                // }
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

        // if (row == 10)
        //     printf("i %d e_left %.3f e_right %.3f e %.3f\n", i, e_left, e_right, e);
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
        //printf("edge_weight[%d] = %.3f\n", i, e_adjust);
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        float e_comm = __shfl_down_sync(0xffffffff, e_sum, offset);
        e_sum += e_comm;
    }

    if (lane_id == 0)
    {
        sum_vec[row] = e_sum;
        // if (row == 10)
        //     printf("row %d len %d e_sum %.3f\n", row, RowPtr[row + 1] - RowPtr[row], e_sum);
    }
}

#define SUBWARP_SIZE 1
#define SUBWARP_NUM (WARP_SIZE / SUBWARP_SIZE)
__global__ void gat_aggregate_kernel_get_sum_subwarp(int m, int nnz, 
int *RowPtr, int *ColIdx, float relu_l, float *a_vec, float *sum_vec, float *edge_weight)
{
    int local_tid = threadIdx.x;
    int local_swid = local_tid / SUBWARP_SIZE;
    int local_wid = local_tid / WARP_SIZE;
    int global_tid = blockIdx.x * blockDim.x + local_tid;
    int global_swid = global_tid / SUBWARP_SIZE;
    int global_wid = global_tid / WARP_SIZE;

    int lane_id = local_tid & (WARP_SIZE - 1);
    int sublane_id = local_tid & (SUBWARP_SIZE - 1);
    int subwarp_id = lane_id / SUBWARP_SIZE;

    unsigned subwarp_mask = 1 >> SUBWARP_SIZE - 1;
    subwarp_mask = subwarp_mask >> (subwarp_id * SUBWARP_SIZE);

    if (global_swid >= m) return;

    int row = global_swid;
    int start_ptr = RowPtr[row];
    int end_ptr = RowPtr[row + 1];

    float e_left = a_vec[row << 1];
    float e_max = 0;

    for (int i = start_ptr + sublane_id; i < end_ptr; i += SUBWARP_SIZE)
    {
        int nid = ColIdx[i];
        float e_right = a_vec[(nid << 1) + 1];
        float e = e_left + e_right;
        e_max = kg_max(kg_max(e, e * relu_l) + REGULAR, e_max);
    }

    unsigned mask = __activemask();
    for (int offset = SUBWARP_SIZE / 2; offset > 0; offset /= 2)
    {
        float e_comm = __shfl_down_sync(mask, e_max, offset);
        e_max = kg_max(e_comm, e_max);
    }

    e_max = __shfl_sync(subwarp_mask, e_max, subwarp_id * SUBWARP_SIZE) - REGULAR;

    float e_sum = 0;

    for (int i = start_ptr + sublane_id; i < end_ptr; i += SUBWARP_SIZE)
    {
        int nid = ColIdx[i];
        float e_right = a_vec[(nid << 1) + 1];
        float e = e_left + e_right;
        float e_adjust = __expf(kg_max(e, e * relu_l) - e_max);

        e_sum += e_adjust;

        edge_weight[i] = e_adjust;
    }

    for (int offset = SUBWARP_SIZE / 2; offset > 0; offset /= 2)
    {
        float e_comm = __shfl_down_sync(mask, e_sum, offset);
        e_sum += e_comm;
    }

    if (sublane_id == 0)
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
    }
}

__global__ void gat_aggregate_new(int m, int nnz, int *RowPtr, int *ColIdx, 
float relu_l, float *a_vec, float *sum_vec, float *edge_weight, block_info2 *binfo)
{
    int bid = blockIdx.x;
    int local_tid = threadIdx.x;
    int local_wid = local_tid / WARP_SIZE;
    int global_tid = blockIdx.x * blockDim.x + local_tid;
    int global_wid = global_tid / WARP_SIZE;
    int lane_id = local_tid & (WARP_SIZE - 1);

    block_info2 info = binfo[bid];

    extern __shared__ int s[];

    int rows = info.row_ed - info.row_st;
    int col_st = RowPtr[info.row_st];
    int col_ed = RowPtr[info.row_ed];
    int nnzs = col_ed - col_st;
    if (nnzs <= SINGLE_PCK_THRESH * 2)
    {
        int *buffer_rowptr = (int*)s;
        // This buffer for nnz-wise scattered values
        float *buffer_1 = (float*)(buffer_rowptr + rows + 1);
        // This buffer for row-wise reduction values
        float *buffer_2 = (float*)(buffer_1 + col_ed - col_st);

        for (int i = info.row_st + local_tid; i <= info.row_ed; i+= blockDim.x)
        {
            buffer_rowptr[i - info.row_st] = RowPtr[i];
            //printf("local_row %d ptr %d\n", i - info.row_st, RowPtr[i]);
            //buffer_rowptr[i - info.row_st] = (buffer_rowptr[i - info.row_st] + WARP_SIZE - 1) / WARP_SIZE;
        }
        __syncthreads();

        // nnz-wise
        int current_row = 0;
        for (int i = col_st + local_tid; i < col_ed; i+= blockDim.x)
        {
            while (buffer_rowptr[current_row + 1] <= i)
                current_row++;

            int mid = info.row_st + current_row;
            int nid = ColIdx[i];
            float e_left = a_vec[mid << 1];
            float e_right = a_vec[(nid << 1) + 1];
            float e = e_left + e_right;
            e = kg_max(e, e * relu_l);

            buffer_1[i - col_st] = e;
            // edge_weight[i] = e;
            // e_max = kg_max(kg_max(e, e * relu_l) + REGULAR, e_max);
            // if (info.row_st + current_row == 10)
            //     printf("i %d src %d dst %d e %.3f\n", i, mid, nid, e);
        }

        __syncthreads();

        // row-wise
        for (int i = info.row_st + local_tid; i < info.row_ed; i+= blockDim.x)
        {
            int crow_st = buffer_rowptr[i - info.row_st];
            int crow_ed = buffer_rowptr[i - info.row_st + 1];

            float e_max = buffer_1[crow_st - col_st];
            for (int j = crow_st + 1; j < crow_ed; j++)
            {
                e_max = kg_max(e_max, buffer_1[j - col_st]);
            }

            buffer_2[i - info.row_st] = e_max;
            // if (i == 10)
            //     printf("row %d e_max %.3f\n", i, e_max);
        }

        __syncthreads();

        // nnz-wise
        current_row = 0;
        for (int i = col_st + local_tid; i < col_ed; i+= blockDim.x)
        {
            while (buffer_rowptr[current_row + 1] <= i)
                current_row++;

            float e = __expf(buffer_1[i - col_st] - buffer_2[current_row]);
            buffer_1[i - col_st] = e;
            edge_weight[i] = e;
            //printf("%d %.3f\n", i, e);
        }

        __syncthreads();

        // row-wise
        for (int i = info.row_st + local_tid; i < info.row_ed; i+= blockDim.x)
        {
            int crow_st = buffer_rowptr[i - info.row_st];
            int crow_ed = buffer_rowptr[i - info.row_st + 1];

            float e_total = 0;
            for (int j = crow_st; j < crow_ed; j++)
            {
                e_total = e_total + buffer_1[j - col_st];
                //printf("j %d e %.3f\n", j, buffer_1[j - col_st]);
            }

            //buffer_2[i - info.row_st] = e_total;
            sum_vec[i] = e_total;
            // if (i < 20)
            //     printf("row %d e_total %.3f\n", i, e_total);
        }
    }
    else
    {
        for (int current_row = info.row_st; current_row < info.row_ed; current_row++)
        {
            col_st = RowPtr[current_row];
            col_ed = RowPtr[current_row + 1];
            float *buffer_reduce = (float*)s;

            float e_max = -1e30;
            float e_left = a_vec[current_row << 1];
            for (int i = col_st + local_tid; i < col_ed; i+= blockDim.x)
            {
                int nid = ColIdx[i];
                float e_right = a_vec[(nid << 1) + 1];
                float e = e_left + e_right;

                e_max = kg_max(kg_max(e, e * relu_l), e_max);
                // if (current_row == m - 1)
                // printf("tid %d i %d e_max %.3f e_left %.3f e_right %.3f\n", local_tid, i, e_max, e_left, e_right);
            }
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
            {
                float e_comm = __shfl_down_sync(0xffffffff, e_max, offset);
                e_max = kg_max(e_comm, e_max);
            }
            if (!lane_id)
                buffer_reduce[local_wid] = e_max;

            __syncthreads();

            for (int i = 0; i < blockDim.x / WARP_SIZE; i++)
                e_max = kg_max(buffer_reduce[i], e_max);

            // if (!local_tid && current_row == m - 1)
            // {
            //     printf("row %d e_max %.3f\n", current_row, e_max);
            // }

            __syncthreads();

            float e_sum = 0;
            for (int i = col_st + local_tid; i < col_ed; i+= blockDim.x)
            {
                int nid = ColIdx[i];
                float e_right = a_vec[(nid << 1) + 1];
                float e = e_left + e_right;

                e = __expf(kg_max(e, e * relu_l) - e_max);
                edge_weight[i] = e;

                // if (current_row == m - 1)
                // printf("left %.3f right %.3f e_max %.3f edge_weight[%d] = %.3f\n", e_left, e_right, e_max, i, e);

                e_sum += e;
            }
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
            {
                float e_comm = __shfl_down_sync(0xffffffff, e_sum, offset);
                e_sum += e_comm;
            }
            if (!lane_id)
                buffer_reduce[local_wid] = e_sum;

            // if (!lane_id && current_row == m - 1)
            // {
            //     printf("row %d e_sum %.3f\n", current_row, e_sum);
            // }

            __syncthreads();
            if (local_tid == 0)
            {
                e_sum = 0;
                for (int i = 0; i < blockDim.x / WARP_SIZE; i++)
                    e_sum += buffer_reduce[i];
                sum_vec[current_row] = e_sum;
            }

            __syncthreads();
        }
        
    }
}

extern void flash_partition_simple(int m, int nnz, int *RowPtr, int *ColIdx, int pck_size,
block_info2 *binfo, int *binfo_n);

void gat_aggregate_balance(int m, int nnz, int feat_len, int *RowPtr, int *ColIdx, 
float *in_feat, float *a_vec, float relu_l, float *out_feat, float *sum_vec, float *edge_weight,
warp_info* winfo, int winfo_n)
{
    printf("--GAT time--\n");

    cudaDeviceSynchronize();

    struct timeval tv_begin, tv_end;
    gettimeofday(&tv_begin, NULL);

    // row-wise subwarp parallel

    // cudaDeviceSynchronize();

    // cudaEvent_t start, stop;
    // float elapsedTime = 0.0;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start, 0);

    // int warp_num = (m + SUBWARP_NUM - 1) / SUBWARP_NUM;
    // const int divide_bs = 128;
    // int thread_num = warp_num * WARP_SIZE;
    // int block_num = (thread_num + divide_bs - 1) / divide_bs;
    // gat_aggregate_kernel_get_sum_subwarp<<<block_num, divide_bs>>>(m, nnz, 
    // RowPtr, ColIdx, relu_l, a_vec, sum_vec, edge_weight);
    // cudaDeviceSynchronize();

    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&elapsedTime, start, stop);

    // --------------------------

    // float n1_host[m], n2_host[m];
    // float t1_host[nnz], t2_host[nnz];

    // row-wise warp parallel

    // cudaDeviceSynchronize();
    // cudaEvent_t start, stop;
    // float elapsedTime = 0.0;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start, 0);

    // int warp_num = m;
    // const int divide_bs = 128;
    // int thread_num = warp_num * WARP_SIZE;
    // int block_num = (thread_num + divide_bs - 1) / divide_bs;
    // gat_aggregate_kernel_get_sum<<<block_num, divide_bs>>>(m, nnz, 
    // RowPtr, ColIdx, relu_l, a_vec, sum_vec, edge_weight);
    // cudaDeviceSynchronize();

    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&elapsedTime, start, stop);

    // cudaMemcpy(n1_host, sum_vec, sizeof(int) * m, cudaMemcpyDeviceToHost);

    // --------------------------

    // nnz-wise balance parallel
    int warp_num, thread_num;
    block_info2 *binfo;
    int *binfo_n;
    cudaMalloc(&binfo, sizeof(block_info2) * m);
    cudaMalloc(&binfo_n, sizeof(int));
    flash_partition_simple(m, nnz, RowPtr, ColIdx, SINGLE_PCK_THRESH, binfo, binfo_n);
    int block_num = 0;
    const int divide_bs = 128;
    cudaMemcpy(&block_num, binfo_n, sizeof(int), cudaMemcpyDeviceToHost);
    int shared_mem_size = SINGLE_PCK_THRESH * 2 * (sizeof(float) * 2 + sizeof(int));

    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    float elapsedTime = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // printf("binfo_n %d\n", block_num);
    gat_aggregate_new<<<block_num, divide_bs, shared_mem_size>>>(m, nnz, RowPtr, ColIdx,
    relu_l, a_vec, sum_vec, edge_weight, binfo);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // cudaMemcpy(n2_host, sum_vec, sizeof(int) * m, cudaMemcpyDeviceToHost);

    // printf("row 10 %.3f %.3f\n", n1_host[10], n2_host[10]);

    // for (int i = 0; i < m; i++)
    // {
    //     if (fabs(n1_host[i] - n2_host[i]) > 0.05)
    //         printf("%d n1 %.3f n2 %.3f\n", i, n1_host[i], n2_host[i]);
    // }
    
    // --------------------------

    // gettimeofday(&tv_end, NULL);

    printf("Attention time: %.2f us\n", elapsedTime * 1000);

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

    printf("Aggregation time: %.2f us\n", duration(tv_begin, tv_end));

    warp_num = m;
    block_num = (warp_num * WARP_SIZE + divide_bs - 1) / divide_bs;

    // gettimeofday(&tv_begin, NULL);

    // // gat_divide_sum<<<block_num, divide_bs>>>(m, feat_len, out_feat, sum_vec);
    // // cudaDeviceSynchronize();

    // gettimeofday(&tv_end, NULL);

    // printf("Divid sum time: %.2f us\n", duration(tv_begin, tv_end));
    // printf("------------\n");
}

__global__ void aggregate_long_row(int feat_len, int *ColIdx, float *feat_in, float *feat_out, float *edge_weight, float *sum_vec,
int current_row, int col_st, int col_ed, int nnz_per_warp)
{
    int blocks = gridDim.x;
    int bid = blockIdx.x;
    int local_tid = threadIdx.x;
    int local_wid = local_tid / WARP_SIZE;
    int lane_id = local_tid & (WARP_SIZE - 1);
    int local_warp_num = blockDim.x / WARP_SIZE;

    col_st = col_st + nnz_per_warp * bid * local_warp_num;
    col_ed = kg_min(col_ed, col_st + nnz_per_warp * local_warp_num);

    // for (int k = lane_id; k < feat_len; k+= WARP_SIZE)
    // {
    //     float aggr_sum = 0;
    //     for (int i = col_st + local_wid; i < col_ed; i+= local_warp_num)
    //     {
    //         int nid = ColIdx[i];
    //         aggr_sum += feat_in[nid * feat_len + k] * edge_weight[i];
    //     }
    //     aggr_sum /= sum_vec[current_row];
    //     atomicAdd(&feat_out[current_row * feat_len + k], aggr_sum);
    // }
}

__global__ void gat_aggregate_all_fused(int m, int nnz, int *RowPtr, int *ColIdx, 
float relu_l, float *a_vec, float *sum_vec, float *edge_weight, 
float *feat_in, float *feat_out, int feat_len, block_info2 *binfo)
{
    int bid = blockIdx.x;
    int local_tid = threadIdx.x;
    int local_wid = local_tid / WARP_SIZE;
    int global_tid = blockIdx.x * blockDim.x + local_tid;
    int global_wid = global_tid / WARP_SIZE;
    int lane_id = local_tid & (WARP_SIZE - 1);
    int local_warp_num = blockDim.x / WARP_SIZE;

    block_info2 info = binfo[bid];

    extern __shared__ int s[];

    int rows = info.row_ed - info.row_st;
    int col_st = RowPtr[info.row_st];
    int col_ed = RowPtr[info.row_ed];
    int nnzs = col_ed - col_st;
    if (nnzs <= SINGLE_PCK_THRESH * 2)
    {
        int *buffer_rowptr = (int*)s;
        // This buffer for nnz-wise scattered values
        float *buffer_1 = (float*)(buffer_rowptr + rows + 1);
        // This buffer for row-wise reduction values
        float *buffer_2 = (float*)(buffer_1 + col_ed - col_st);

        for (int i = info.row_st + local_tid; i <= info.row_ed; i+= blockDim.x)
        {
            buffer_rowptr[i - info.row_st] = RowPtr[i];
        }
        __syncthreads();

        // nnz-wise
        int current_row = 0;
        for (int i = col_st + local_tid; i < col_ed; i+= blockDim.x)
        {
            while (buffer_rowptr[current_row + 1] <= i)
                current_row++;

            int mid = info.row_st + current_row;
            int nid = ColIdx[i];
            float e_left = a_vec[mid << 1];
            float e_right = a_vec[(nid << 1) + 1];
            float e = e_left + e_right;
            e = kg_max(e, e * relu_l);

            buffer_1[i - col_st] = e;
        }

        __syncthreads();

        // row-wise
        for (int i = info.row_st + local_tid; i < info.row_ed; i+= blockDim.x)
        {
            int crow_st = buffer_rowptr[i - info.row_st];
            int crow_ed = buffer_rowptr[i - info.row_st + 1];

            float e_max = buffer_1[crow_st - col_st];
            for (int j = crow_st + 1; j < crow_ed; j++)
            {
                e_max = kg_max(e_max, buffer_1[j - col_st]);
            }

            buffer_2[i - info.row_st] = e_max;
        }

        __syncthreads();

        // nnz-wise
        current_row = 0;
        for (int i = col_st + local_tid; i < col_ed; i+= blockDim.x)
        {
            while (buffer_rowptr[current_row + 1] <= i)
                current_row++;

            float e = __expf(buffer_1[i - col_st] - buffer_2[current_row]);
            buffer_1[i - col_st] = e;
            edge_weight[i] = e;
        }

        __syncthreads();

        // row-wise
        for (int i = info.row_st + local_tid; i < info.row_ed; i+= blockDim.x)
        {
            int crow_st = buffer_rowptr[i - info.row_st];
            int crow_ed = buffer_rowptr[i - info.row_st + 1];

            float e_total = 0;
            for (int j = crow_st; j < crow_ed; j++)
            {
                e_total = e_total + buffer_1[j - col_st];
            }

            buffer_2[i - info.row_st] = e_total;
            sum_vec[i] = e_total;
        }

        __syncthreads();

        int nnz_per_warp = (nnzs + local_warp_num - 1) / local_warp_num;
        // local
        int aggr_col_st = nnz_per_warp * local_wid;
        int aggr_col_ed = aggr_col_st + nnz_per_warp;
        // local
        int aggr_row_st = 0, aggr_row_ed = 0;

        // if (!local_tid)
        // {
        //     for (int i = 0; i <= info.row_ed - info.row_st; i++)
        //         printf("%d ", buffer_rowptr[i]);
        //     printf("\n");
        // }

        while (aggr_row_st < info.row_ed - info.row_st - 1 && col_st + aggr_col_st > buffer_rowptr[aggr_row_st + 1])
        {
            aggr_row_st++;
        }
        aggr_row_ed = aggr_row_st;
        while (aggr_row_ed < info.row_ed - info.row_st && col_st + aggr_col_ed > buffer_rowptr[aggr_row_ed])
        {
            aggr_row_ed++;
        }

        // if (!lane_id)
        //     printf("warp_id %d row %d %d col %d %d nnz %d nnz_per_warp %d\n", local_wid, aggr_row_st, aggr_row_ed, aggr_col_st, aggr_col_ed,
        //     nnz, nnz_per_warp);
        
        if (col_st + aggr_col_st < col_ed)
        {
            // if (!lane_id)
            //     printf("wid %d %d\n", aggr_col_st, col_st);
            // return;
            for (int i = aggr_row_st; i < aggr_row_ed; i++)
            {
                int current_col_st = buffer_rowptr[i];
                int current_col_ed = buffer_rowptr[i + 1];
                if (i == aggr_row_st)
                    current_col_st = col_st + aggr_col_st;
                if (i == aggr_row_ed - 1)
                    current_col_ed = kg_min(col_st + aggr_col_ed, col_ed);

                // if (!lane_id)
                //     printf("wid %d work on %d st %d ed %d\n", local_wid, i, current_col_st, current_col_ed);

                for (int k = lane_id; k < feat_len; k+= WARP_SIZE)
                {
                    float aggr_sum = 0.0;
                    for (int j = current_col_st; j < current_col_ed; j++)
                    {
                        int nid = ColIdx[j];

                        //aggr_sum += edge_weight[j] * feat_in[nid * feat_len + k];
                        aggr_sum += buffer_1[j - col_st] * feat_in[nid * feat_len + k];

                        // if (i == 0 && !lane_id)
                        //     printf("%d %.3f %.3f\n", nid, buffer_1[j - col_st], feat_in[nid * feat_len + k]);
                    }
                    aggr_sum /= buffer_2[i];
                    atomicAdd(&feat_out[(info.row_st + i) * feat_len + k], aggr_sum);
                    // if (!lane_id)
                    //     printf("wid %d row %d k %d result %.3f\n", local_wid, info.row_st + i, k, feat_out[(info.row_st + i) * feat_len + k]);
                }
            }
        }
    }
    else
    {
        for (int current_row = info.row_st; current_row < info.row_ed; current_row++)
        {
            col_st = RowPtr[current_row];
            col_ed = RowPtr[current_row + 1];
            float *buffer_reduce = (float*)s;

            float e_max = -1e30;
            float e_left = a_vec[current_row << 1];
            for (int i = col_st + local_tid; i < col_ed; i+= blockDim.x)
            {
                int nid = ColIdx[i];
                float e_right = a_vec[(nid << 1) + 1];
                float e = e_left + e_right;

                e_max = kg_max(kg_max(e, e * relu_l), e_max);
                // if (current_row == m - 1)
                // printf("tid %d i %d e_max %.3f e_left %.3f e_right %.3f\n", local_tid, i, e_max, e_left, e_right);
            }
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
            {
                float e_comm = __shfl_down_sync(0xffffffff, e_max, offset);
                e_max = kg_max(e_comm, e_max);
            }
            if (!lane_id)
                buffer_reduce[local_wid] = e_max;

            __syncthreads();

            for (int i = 0; i < blockDim.x / WARP_SIZE; i++)
                e_max = kg_max(buffer_reduce[i], e_max);

            __syncthreads();

            float e_sum = 0;
            for (int i = col_st + local_tid; i < col_ed; i+= blockDim.x)
            {
                int nid = ColIdx[i];
                float e_right = a_vec[(nid << 1) + 1];
                float e = e_left + e_right;

                e = __expf(kg_max(e, e * relu_l) - e_max);
                edge_weight[i] = e;

                e_sum += e;
            }
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
            {
                float e_comm = __shfl_down_sync(0xffffffff, e_sum, offset);
                e_sum += e_comm;
            }
            if (!lane_id)
                buffer_reduce[local_wid] = e_sum;

            __syncthreads();
            if (local_tid == 0)
            {
                e_sum = 0;
                for (int i = 0; i < blockDim.x / WARP_SIZE; i++)
                    e_sum += buffer_reduce[i];
                sum_vec[current_row] = e_sum;
            }

            __syncthreads();

            // for (int k = lane_id; k < feat_len; k+= WARP_SIZE)
            // {
            //     float aggr_sum = 0;
            //     for (int i = col_st + local_wid; i < col_ed; i+= local_warp_num)
            //     {
            //         int nid = ColIdx[i];
            //         aggr_sum += feat_in[nid * feat_len + k] * edge_weight[i];
            //     }
            //     aggr_sum /= sum_vec[current_row];
            //     atomicAdd(&feat_out[current_row * feat_len + k], aggr_sum);
            // }
        //     if (local_tid == 0)
        //     {
        //         int nnz_per_warp = 128;
        //         const int dynamic_bsize = 128;
        //         int nnz_per_block = nnz_per_warp * dynamic_bsize / WARP_SIZE;
        //         int dynamic_blocks = (nnzs + nnz_per_block - 1) / nnz_per_block;
        //         aggregate_long_row<<<dynamic_blocks, dynamic_bsize>>>(feat_len, ColIdx, feat_in, feat_out, edge_weight, sum_vec,
        //         current_row, col_st, col_ed, nnz_per_warp);
        //     }
        //     __syncthreads();
        }
        
    }
}

void gat_aggregate_balance2(int m, int nnz, int feat_len, int *RowPtr, int *ColIdx, 
float *in_feat, float *a_vec, float relu_l, float *out_feat, float *sum_vec, float *edge_weight,
warp_info* winfo, int winfo_n)
{
    printf("--GAT time--\n");

    cudaDeviceSynchronize();

    // row-panel parallel
    int warp_num, thread_num;
    block_info2 *binfo;
    int *binfo_n;
    cudaMalloc(&binfo, sizeof(block_info2) * m);
    cudaMalloc(&binfo_n, sizeof(int));
    flash_partition_simple(m, nnz, RowPtr, ColIdx, SINGLE_PCK_THRESH, binfo, binfo_n);
    int block_num = 0;
    const int divide_bs = 128;
    cudaMemcpy(&block_num, binfo_n, sizeof(int), cudaMemcpyDeviceToHost);
    int shared_mem_size = SINGLE_PCK_THRESH * (sizeof(float) * 2 + sizeof(int));

    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    float elapsedTime = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    gat_aggregate_all_fused<<<block_num, divide_bs, shared_mem_size>>>(m, nnz, RowPtr, ColIdx,
    relu_l, a_vec, sum_vec, edge_weight, in_feat, out_feat, feat_len, binfo);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("Fused GAT time: %.2f us\n", elapsedTime * 1000);

    warp_num = m;
    block_num = (warp_num * WARP_SIZE + divide_bs - 1) / divide_bs;

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
