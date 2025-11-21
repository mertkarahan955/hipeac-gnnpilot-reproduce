#include "preprocessing.h"
#include <cuda.h>
#include <stdio.h>
#include <vector>
#include <algorithm>

#define kg_min(a, b) ((a) < (b) ? (a) : (b))
#define kg_max(a, b) ((a) > (b) ? (a) : (b))

// Simple partition function for row panels (hetero+ kernels)
void partition_row_panels(int m, int nnz, int *RowPtr, int wsize, 
                          std::vector<row_panel> &host_rp) {
    int alpha = 2;  // Approximation factor
    int group_n = 0;
    int last_start_row = 0;
    int last_end_row = -1;
    int last_start_col, last_end_col;

    for (int row = 0; row < m; row++) {
        int row_st = RowPtr[row];
        int row_ed = RowPtr[row + 1];

        if (row_ed - row_st + alpha > wsize - group_n || last_end_row == -1) {
            if (last_end_row != -1) {
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
    kg_info* info = new kg_info();
    
    int wsize = 32;  // Warp size
    
    // Partition for row panels (hetero+ kernels)
    std::vector<row_panel> host_rp;
    partition_row_panels(m, nnz, RowPtr, wsize, host_rp);
    info->rp_n_host = host_rp.size();
    
    // Always allocate, even if empty (kernel expects valid pointers)
    if (info->rp_n_host == 0) {
        info->rp_n_host = 1;  // At least one dummy entry
        host_rp.push_back(row_panel(0, 0, 0, 0));
    }
    cudaMalloc(&info->rp_info, host_rp.size() * sizeof(row_panel));
    cudaMalloc(&info->rp_n, sizeof(int));
    cudaMemcpy(info->rp_info, host_rp.data(), host_rp.size() * sizeof(row_panel), 
               cudaMemcpyHostToDevice);
    int rp_n = info->rp_n_host;
    cudaMemcpy(info->rp_n, &rp_n, sizeof(int), cudaMemcpyHostToDevice);
    
    // Partition for edge panels (edge-parallel kernels)
    std::vector<row_panel> host_ep;
    partition_edge_panels(m, nnz, RowPtr, wsize, host_ep);
    info->ep_n_host = host_ep.size();
    
    // Always allocate, even if empty (kernel expects valid pointers)
    if (info->ep_n_host == 0) {
        info->ep_n_host = 1;  // At least one dummy entry
        host_ep.push_back(row_panel(0, 0, 0, 0));
    }
    cudaMalloc(&info->ep_info, host_ep.size() * sizeof(row_panel));
    cudaMalloc(&info->ep_n, sizeof(int));
    cudaMemcpy(info->ep_info, host_ep.data(), host_ep.size() * sizeof(row_panel), 
               cudaMemcpyHostToDevice);
    int ep_n = info->ep_n_host;
    cudaMemcpy(info->ep_n, &ep_n, sizeof(int), cudaMemcpyHostToDevice);
    
    // Partition for neighbor groups (ngd kernels)
    std::vector<neighbor_group> host_ng;
    partition_neighbor_groups(m, nnz, RowPtr, host_ng);
    info->ng_n_host = host_ng.size();
    
    // Always allocate, even if empty (kernel expects valid pointers)
    if (info->ng_n_host == 0) {
        info->ng_n_host = 1;  // At least one dummy entry
        host_ng.push_back(neighbor_group(0, 0));
    }
    cudaMalloc(&info->ng_info, host_ng.size() * sizeof(neighbor_group));
    cudaMalloc(&info->ng_n, sizeof(int));
    cudaMemcpy(info->ng_info, host_ng.data(), host_ng.size() * sizeof(neighbor_group), 
               cudaMemcpyHostToDevice);
    int ng_n = info->ng_n_host;
    cudaMemcpy(info->ng_n, &ng_n, sizeof(int), cudaMemcpyHostToDevice);
    
    return (int64_t)info;
}

