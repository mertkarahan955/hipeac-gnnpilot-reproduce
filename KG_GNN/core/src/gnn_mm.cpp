#include "../include/KG_GNN.h"
#include <torch/extension.h>
#include <stdio.h>

extern void kg_mm_execute(int m, int n, int k, float *input_a, float *input_b, float *output_c);
extern void cublas_mm_execute(int m, int n, int k, float *input_a, float *input_b, float *output_c);
extern void kg_mm_pipeline_execute(int m, int n, int k, float *input_a, float *input_b, float *output_c);
extern void kg_spmm_mm_pipeline_execute(int m, int n, int k, int *rowptr, int *colidx, float *input_a, float *input_b, float *output_c);
extern void kg_spmm_mm_pipeline_balance_execute(int m, int n, int k, int *rowptr, int *colidx, 
float *input_a, float *input_b, float *output_c, warp_info *winfo, int winfo_n);
extern void kg_spmm_mm_pipeline_final(int m, int n, int k, int *rowptr, int *colidx, 
float *input_a, float *input_b, float *interm_c, float *output_c, warp_info *winfo, int winfo_n);

extern void kg_spmm_mm_balance(int m, int nnz, int *RowPtr, warp_info **winfo, int *winfo_n);
extern void kg_spmm_mm_balance_final(int m, int nnz, int *RowPtr, warp_info **winfo, int *winfo_n);

void kg_gcn_run_mm(
    torch::Tensor input_a,
    torch::Tensor input_b,
    torch::Tensor output_c
)
{
    int m = input_a.size(0);
    int n = input_b.size(1);
    int k = input_a.size(1);

    kg_mm_execute(m, n, k, input_a.data_ptr<float>(), input_b.data_ptr<float>(), output_c.data_ptr<float>());
}

void cublas_run_mm(
    torch::Tensor input_a,
    torch::Tensor input_b,
    torch::Tensor output_c
)
{
    int m = input_a.size(0);
    int n = input_b.size(1);
    int k = input_a.size(1);

    cublas_mm_execute(m, n, k, input_a.data_ptr<float>(), input_b.data_ptr<float>(), output_c.data_ptr<float>());
}

void kg_gcn_run_fused_mm(
    torch::Tensor input_a,
    torch::Tensor input_b,
    torch::Tensor output_c
)
{
    int m = input_a.size(0);
    int n = input_b.size(1);
    int k = input_a.size(1);

    kg_mm_pipeline_execute(m, n, k, input_a.data_ptr<float>(), input_b.data_ptr<float>(), output_c.data_ptr<float>());
}

void kg_gcn_run_fused_spmm_mm(
    torch::Tensor RowPtr,
    torch::Tensor ColIdx,
    torch::Tensor input_a,
    torch::Tensor input_b,
    torch::Tensor output_c
)
{
    int m = input_a.size(0);
    int n = input_b.size(1);
    int k = input_a.size(1);

    kg_spmm_mm_pipeline_execute(m, n, k, 
    RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(),
    input_a.data_ptr<float>(), input_b.data_ptr<float>(), output_c.data_ptr<float>());
}

int64_t kg_gcn_spmm_mm_balance(
    torch::Tensor RowPtr,
    torch::Tensor ColIdx)
{
    int m = RowPtr.size(0) - 1;
    int nnz = ColIdx.size(0);
    int winfo_n;
    warp_info *winfo;
    kg_spmm_mm_balance(m, nnz, RowPtr.data_ptr<int>(), &winfo, &winfo_n);
    ana_info *ret = new ana_info(winfo, winfo_n);
    return (int64_t)ret;
}

int64_t kg_gcn_spmm_mm_balance_final(
    torch::Tensor RowPtr,
    torch::Tensor ColIdx)
{
    int m = RowPtr.size(0) - 1;
    int nnz = ColIdx.size(0);
    int winfo_n;
    warp_info *winfo;
    kg_spmm_mm_balance_final(m, nnz, RowPtr.data_ptr<int>(), &winfo, &winfo_n);
    ana_info *ret = new ana_info(winfo, winfo_n);
    return (int64_t)ret;
}

void kg_gcn_spmm_mm_run_balance(
    torch::Tensor RowPtr,
    torch::Tensor ColIdx,
    torch::Tensor input_a,
    torch::Tensor input_b,
    torch::Tensor output_c,
    int64_t ana_add
)
{
    int m = RowPtr.size(0) - 1;
    int n = input_b.size(1);
    int k = input_a.size(1);
    // int nnz = ColIdx.size(0);
    // int feat_len = input_a.size(1);

    ana_info ana = *((ana_info*)ana_add);

    kg_spmm_mm_pipeline_balance_execute(m, n, k, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(),
    input_a.data_ptr<float>(), input_b.data_ptr<float>(), output_c.data_ptr<float>(), (warp_info*)ana.winfo, ana.winfo_n);

}

void kg_gcn_spmm_mm_run_final(
    torch::Tensor RowPtr,
    torch::Tensor ColIdx,
    torch::Tensor input_a,
    torch::Tensor input_b,
    torch::Tensor interm_c,
    torch::Tensor output_c,
    int64_t ana_add
)
{
    int m = RowPtr.size(0) - 1;
    int n = input_b.size(1);
    int k = input_a.size(1);
    // int nnz = ColIdx.size(0);
    // int feat_len = input_a.size(1);

    ana_info ana = *((ana_info*)ana_add);

    kg_spmm_mm_pipeline_final(m, n, k, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(),
    input_a.data_ptr<float>(), input_b.data_ptr<float>(), 
    interm_c.data_ptr<float>(), output_c.data_ptr<float>(),
    (warp_info*)ana.winfo, ana.winfo_n);

}

