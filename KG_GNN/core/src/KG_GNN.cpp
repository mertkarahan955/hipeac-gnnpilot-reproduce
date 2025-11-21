#include "../include/KG_GNN.h"
#include <torch/extension.h>
#include <stdio.h>

extern int64_t kg_gcn_balance(
    torch::Tensor RowPtr,
    torch::Tensor ColIdx,
    int64_t wsize
);

extern int64_t kg_gcn_balance2(
    torch::Tensor RowPtr,
    torch::Tensor ColIdx,
    int64_t wsize,
    int64_t alpha
);

extern int64_t kg_gcn_balance3(
    torch::Tensor RowPtr,
    torch::Tensor ColIdx,
    int64_t wsize,
    int64_t alpha
);

extern int64_t kg_gcn_balance4(
    torch::Tensor RowPtr,
    torch::Tensor ColIdx,
    int64_t wsize
);

extern int64_t kg_gcn_schedule_locality(
    torch::Tensor RowPtr,
    torch::Tensor ColIdx,
    int bin_size
);

extern int64_t kg_gcn_block_schedule(
    torch::Tensor RowPtr,
    torch::Tensor ColIdx
);

extern int64_t kg_gcn_bin_pack(
    torch::Tensor RowPtr, 
    torch::Tensor ColIdx,
    int64_t bin_size,
    int64_t pack_size,
    int64_t bin_thresh,
    int64_t bin_block,
    int64_t wsize,
    int64_t alpha
);

extern void kg_gcn_run(
    torch::Tensor RowPtr,
    torch::Tensor ColIdx,
    torch::Tensor in_feat,
    torch::Tensor out_feat
);


extern void kg_gcn_run_balance(
    torch::Tensor RowPtr,
    torch::Tensor ColIdx,
    torch::Tensor in_feat,
    torch::Tensor out_feat,
    int64_t ana_add
);

extern float kg_gcn_run_cusparse(
    torch::Tensor RowPtr,
    torch::Tensor ColIdx,
    torch::Tensor Values,
    torch::Tensor in_feat,
    torch::Tensor out_feat,
    int warmup,
    int repetitions
);

extern void kg_gin_run_balance(
    torch::Tensor RowPtr,
    torch::Tensor ColIdx,
    torch::Tensor in_feat,
    torch::Tensor out_feat,
    float eps,
    int64_t ana_add
);

void kg_gcn_run_balance_with_deg(
    torch::Tensor RowPtr,
    torch::Tensor ColIdx,
    torch::Tensor in_feat,
    torch::Tensor out_feat,
    torch::Tensor degree,
    int64_t ana_add
);

extern void kg_gcn_run_balance_shared(
    torch::Tensor RowPtr,
    torch::Tensor ColIdx,
    torch::Tensor in_feat,
    torch::Tensor out_feat,
    int64_t ana_add
);

extern void kg_gcn_run_mm(
    torch::Tensor input_a,
    torch::Tensor input_b,
    torch::Tensor output_c
);

extern void cublas_run_mm(
    torch::Tensor input_a,
    torch::Tensor input_b,
    torch::Tensor output_c
);

extern void kg_gcn_run_fused_mm(
    torch::Tensor input_a,
    torch::Tensor input_b,
    torch::Tensor output_c
);

extern void kg_gcn_run_fused_spmm_mm(
    torch::Tensor RowPtr,
    torch::Tensor ColIdx,
    torch::Tensor input_a,
    torch::Tensor input_b,
    torch::Tensor output_c
);

extern void kg_gcn_spmm_mm_run_balance(
    torch::Tensor RowPtr,
    torch::Tensor ColIdx,
    torch::Tensor input_a,
    torch::Tensor input_b,
    torch::Tensor output_c,
    int64_t ana_add
);

extern void kg_sddmm_run_balance(
    torch::Tensor RowPtr,
    torch::Tensor ColIdx,
    torch::Tensor in_feat1,
    torch::Tensor in_feat2,
    torch::Tensor out_feat,
    int64_t ana_add
);

extern float kg_sddmm_run_cusparse(
    torch::Tensor RowPtr,
    torch::Tensor ColIdx,
    torch::Tensor in_feat1,
    torch::Tensor in_feat2,
    int feat_len,
    torch::Tensor out_feat,
    int warmup,
    int repetitions
);

void kg_gat_run_balance(
    torch::Tensor RowPtr,
    torch::Tensor ColIdx,
    torch::Tensor in_feat,
    torch::Tensor out_feat,
    torch::Tensor a_weight,
    torch::Tensor sum_vec,
    torch::Tensor edge_weight,
    double relu_l,
    int64_t ana_add
);

extern void kg_gcn_finalize(
    int64_t ana_add
);

// TORCH_LIBRARY(KGGNN, m) {
//     m.def("kg_gcn_run", &kg_gcn_run, "KGGNN GCN run");
//     m.def("kg_gcn_run_balance", &kg_gcn_run_balance, "KGGNN GCN run balance");
//     m.def("kg_gcn_run_balance_with_deg", &kg_gcn_run_balance_with_deg, "KGGNN GCN run balance");
//     m.def("kg_gcn_run_balance_shared", &kg_gcn_run_balance_shared, "KGGNN GCN run balance with shared memory");
//     m.def("kg_nn_gcn_fused_run", &kg_nn_gcn_fused_run, "KGGNN GCN run balance");
//     m.def("kg_gcn_spmm_mm_balance", &kg_gcn_spmm_mm_balance, "KGGNN GCN spmm + mm run balance");
//     m.def("kg_gcn_spmm_mm_balance_final", &kg_gcn_spmm_mm_balance_final, "KGGNN GCN spmm + mm run balance final");
//     m.def("kg_gcn_spmm_mm_run_final", &kg_gcn_spmm_mm_run_final, "KGGNN GCN spmm + mm run final");

//     m.def("kg_gat_run_balance", &kg_gat_run_balance, "KGGNN GAT run balance");
//     m.def("kg_gin_run_balance", &kg_gin_run_balance, "KGGNN GAT run balance");

//     m.def("kg_sddmm_run_balance", &kg_sddmm_run_balance, "KGGNN SDDMM run balance");

//     m.def("kg_gcn_run_cusparse", &kg_gcn_run_cusparse, "KGGNN GCN run cuSPARSE");
//     m.def("kg_sddmm_run_cusparse", &kg_sddmm_run_cusparse, "KGGNN SDDMM run cuSPARSE");

//     m.def("kg_gcn_run_mm", &kg_gcn_run_mm, "KGGNN GCN run matrix multiplication");
//     m.def("cublas_run_mm", &cublas_run_mm, "cuBLAS run matrix multiplication");
//     m.def("kg_gcn_run_fused_mm", &kg_gcn_run_fused_mm, "KGGNN fused matrix multiplication");
//     m.def("kg_gcn_run_fused_spmm_mm", &kg_gcn_run_fused_spmm_mm, "KGGNN fused SpMM + matrix multiplication");
//     m.def("kg_gcn_spmm_mm_run_balance", &kg_gcn_spmm_mm_run_balance, "EFGNN: fused SpMM + matrix multiplication");

//     m.def("kg_gcn_balance", &kg_gcn_balance, "KGGNN GCN balance");
//     m.def("kg_gcn_balance2", &kg_gcn_balance2, "KGGNN GCN balance2");
//     m.def("kg_gcn_balance3", &kg_gcn_balance3, "KGGNN GCN balance3");
//     m.def("kg_gcn_balance4", &kg_gcn_balance4, "KGGNN GCN balance4");
//     m.def("kg_gcn_schedule_locality", &kg_gcn_schedule_locality, "KGGNN GCN schedule locality");
//     m.def("kg_gcn_block_schedule", &kg_gcn_block_schedule, "KGGNN GCN block schedule");
//     m.def("kg_gcn_bin_pack", &kg_gcn_bin_pack, "KGGNN GCN block schedule");

//     m.def("kg_gcn_finalize", &kg_gcn_finalize, "KGGNN GCN finalize");
// }

TORCH_LIBRARY(KGGNN, m) {
    m.def("kg_gcn_run", &kg_gcn_run);
    m.def("kg_gcn_run_balance", &kg_gcn_run_balance);
    m.def("kg_gcn_run_balance_with_deg", &kg_gcn_run_balance_with_deg);
    m.def("kg_gcn_run_balance_shared", &kg_gcn_run_balance_shared);

    m.def("kg_gat_run_balance", &kg_gat_run_balance);
    // m.def("kg_gin_run_balance", &kg_gin_run_balance);

    // m.def("kg_sddmm_run_balance", &kg_sddmm_run_balance);

    // m.def("kg_gcn_run_cusparse", &kg_gcn_run_cusparse);
    // m.def("kg_sddmm_run_cusparse", &kg_sddmm_run_cusparse);

    // m.def("kg_gcn_run_mm", &kg_gcn_run_mm);
    // m.def("cublas_run_mm", &cublas_run_mm);
    // m.def("kg_gcn_run_fused_mm", &kg_gcn_run_fused_mm);
    // m.def("kg_gcn_run_fused_spmm_mm", &kg_gcn_run_fused_spmm_mm);
    // m.def("kg_gcn_spmm_mm_run_balance", &kg_gcn_spmm_mm_run_balance);

    m.def("kg_gcn_balance", &kg_gcn_balance);
    m.def("kg_gcn_balance2", &kg_gcn_balance2);
    m.def("kg_gcn_balance3", &kg_gcn_balance3);
    m.def("kg_gcn_balance4", &kg_gcn_balance4);
    // m.def("kg_gcn_schedule_locality", &kg_gcn_schedule_locality);
    // m.def("kg_gcn_block_schedule", &kg_gcn_block_schedule);
    m.def("kg_gcn_bin_pack", &kg_gcn_bin_pack);

    m.def("kg_gcn_finalize", &kg_gcn_finalize);
}