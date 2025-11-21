#include "preprocessing.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <algorithm>

#define kg_min(a, b) ((a) < (b) ? (a) : (b))
#define kg_max(a, b) ((a) > (b) ? (a) : (b))

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Simple partition function for row panels (hetero+ kernels)
void partition_row_panels(int m, int nnz, int *RowPtr, int wsize, 
                          std::vector<row_panel> &host_rp) {
    int alpha = 2;  // Approximation factor
    int group_n = 0;
    int last_start_row = 0;
    int last_end_row = -1;
    int last_start_col, last_end_col;
    fprintf(stderr, "DEBUG partition_row_panels: Starting partitioning with m=%d, nnz=%d, wsize=%d\n", m, nnz, wsize);
    fflush(stderr);

    for (int row = 0; row < m; row++) {
        int row_st = RowPtr[row];
        int row_ed = RowPtr[row + 1];
        fprintf(stderr, "DEBUG partition_row_panels: row=%d, row_st=%d, row_ed=%d\n", row, row_st, row_ed);
        fflush(stderr);

        if (row_ed - row_st + alpha > wsize - group_n || last_end_row == -1) {
            if (last_end_row != -1) {
                fprintf(stderr, "DEBUG partition_row_panels: Creating panel from (%d, %d) to (%d, %d)\n", 
                        last_start_row, last_start_col, last_end_row, last_end_col);
                fflush(stderr);
                row_panel tmp(last_start_row, last_end_row, last_start_col, last_end_col);
                host_rp.push_back(tmp);
                group_n = 0;
            }

            int wi;
            for (wi = row_st; wi < row_ed - wsize + alpha; wi += wsize - alpha) {
                row_panel tmp(row, row + 1, wi, wi + wsize - alpha);
                host_rp.push_back(tmp);
            }
            last_start_row = row;
            last_start_col = wi;
            group_n += row_ed - wi + alpha;
        } else {
            group_n += row_ed - row_st + alpha;
        }

        last_end_row = row + 1;
        last_end_col = row_ed;
    }
    fprintf(stderr, "DEBUG partition_row_panels: Created final panel from (%d, %d) to (%d, %d)\n", 
            last_start_row, last_start_col, last_end_row, last_end_col);
    fflush(stderr);
    if (last_end_row != -1) {
        row_panel tmp(last_start_row, last_end_row, last_start_col, last_end_col);
        host_rp.push_back(tmp);
    }
}

// Partition for edge panels (edge-parallel kernels)
void partition_edge_panels(int m, int nnz, int *RowPtr, int wsize,
                           std::vector<row_panel> &host_ep) {
    // Similar to row panels but optimized for edge parallelism
    partition_row_panels(m, nnz, RowPtr, wsize, host_ep);
}

// Partition for neighbor groups (ngd kernels)
void partition_neighbor_groups(int m, int nnz, int *RowPtr, 
                               std::vector<neighbor_group> &host_ng) {
    for (int row = 0; row < m; row++) {
        int row_st = RowPtr[row];
        int row_ed = RowPtr[row + 1];
        
        // Partition each row's neighbors into groups
        for (int col = row_st; col < row_ed; col += NG_SIZE) {
            neighbor_group ng(row, col);
            host_ng.push_back(ng);
        }
    }
}

// Main preprocessing function
int64_t preprocessing_cuda(int m, int nnz, int *RowPtr, int *ColIdx, bool long_dynamic) {
    fprintf(stderr, "DEBUG preprocessing_cuda: m=%d, nnz=%d\n", m, nnz);
    fflush(stderr);
    
    kg_info* info = new kg_info();
    if (!info) {
        fprintf(stderr, "ERROR: Failed to allocate kg_info\n");
        fflush(stderr);
        return 0;
    }
    fprintf(stderr, "DEBUG: Allocated kg_info at %p\n", (void*)info);
    fflush(stderr);
    
    int wsize = 32;  // Warp size
    
    // Partition for row panels (hetero+ kernels)
    fprintf(stderr, "DEBUG 1: Partitioning row panels...\n");
    std::vector<row_panel> host_rp;
    fprintf(stderr, "DEBUG 2: Calling partition_row_panels...\n");
    partition_row_panels(m, nnz, RowPtr, wsize, host_rp);
    fprintf(stderr, "DEBUG 3: Returned from partition_row_panels...\n");
    info->rp_n_host = host_rp.size();
    
    // Always allocate, even if empty (kernel expects valid pointers)
    if (info->rp_n_host == 0) {
        info->rp_n_host = 1;  // At least one dummy entry
        host_rp.push_back(row_panel(0, 0, 0, 0));
    }
    CUDA_CHECK(cudaMalloc(&info->rp_info, host_rp.size() * sizeof(row_panel)));
    CUDA_CHECK(cudaMalloc(&info->rp_n, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(info->rp_info, host_rp.data(), host_rp.size() * sizeof(row_panel), 
               cudaMemcpyHostToDevice));
    int rp_n = info->rp_n_host;
    CUDA_CHECK(cudaMemcpy(info->rp_n, &rp_n, sizeof(int), cudaMemcpyHostToDevice));
    
    // Partition for edge panels (edge-parallel kernels)
    std::vector<row_panel> host_ep;
    partition_edge_panels(m, nnz, RowPtr, wsize, host_ep);
    info->ep_n_host = host_ep.size();
    
    // Always allocate, even if empty (kernel expects valid pointers)
    if (info->ep_n_host == 0) {
        info->ep_n_host = 1;  // At least one dummy entry
        host_ep.push_back(row_panel(0, 0, 0, 0));
    }
    CUDA_CHECK(cudaMalloc(&info->ep_info, host_ep.size() * sizeof(row_panel)));
    CUDA_CHECK(cudaMalloc(&info->ep_n, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(info->ep_info, host_ep.data(), host_ep.size() * sizeof(row_panel), 
               cudaMemcpyHostToDevice));
    int ep_n = info->ep_n_host;
    CUDA_CHECK(cudaMemcpy(info->ep_n, &ep_n, sizeof(int), cudaMemcpyHostToDevice));
    
    // Partition for neighbor groups (ngd kernels)
    std::vector<neighbor_group> host_ng;
    partition_neighbor_groups(m, nnz, RowPtr, host_ng);
    info->ng_n_host = host_ng.size();
    
    // Always allocate, even if empty (kernel expects valid pointers)
    if (info->ng_n_host == 0) {
        info->ng_n_host = 1;  // At least one dummy entry
        host_ng.push_back(neighbor_group(0, 0));
    }
    CUDA_CHECK(cudaMalloc(&info->ng_info, host_ng.size() * sizeof(neighbor_group)));
    CUDA_CHECK(cudaMalloc(&info->ng_n, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(info->ng_info, host_ng.data(), host_ng.size() * sizeof(neighbor_group), 
               cudaMemcpyHostToDevice));
    int ng_n = info->ng_n_host;
    CUDA_CHECK(cudaMemcpy(info->ng_n, &ng_n, sizeof(int), cudaMemcpyHostToDevice));
    
    // Synchronize to ensure all operations complete
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Debug: verify all pointers are valid
    fprintf(stderr, "DEBUG: Verifying pointers...\n");
    fprintf(stderr, "rp_info: %p, rp_n: %p, rp_n_host: %d\n", 
            (void*)info->rp_info, (void*)info->rp_n, info->rp_n_host);
    fprintf(stderr, "ep_info: %p, ep_n: %p, ep_n_host: %d\n", 
            (void*)info->ep_info, (void*)info->ep_n, info->ep_n_host);
    fprintf(stderr, "ng_info: %p, ng_n: %p, ng_n_host: %d\n", 
            (void*)info->ng_info, (void*)info->ng_n, info->ng_n_host);
    fflush(stderr);
    
    if (!info->rp_info || !info->rp_n || !info->ep_info || !info->ep_n || 
        !info->ng_info || !info->ng_n) {
        fprintf(stderr, "ERROR: Some pointers are null after allocation!\n");
        fprintf(stderr, "rp_info: %p, rp_n: %p\n", (void*)info->rp_info, (void*)info->rp_n);
        fprintf(stderr, "ep_info: %p, ep_n: %p\n", (void*)info->ep_info, (void*)info->ep_n);
        fprintf(stderr, "ng_info: %p, ng_n: %p\n", (void*)info->ng_info, (void*)info->ng_n);
        fflush(stderr);
        exit(1);
    }
    
    int64_t result = (int64_t)info;
    fprintf(stderr, "DEBUG: preprocessing_cuda returning %lld (0x%llx)\n", 
            (long long)result, (unsigned long long)result);
    fflush(stderr);
    return result;
}

