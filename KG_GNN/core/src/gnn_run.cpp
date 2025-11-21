#include "../include/KG_GNN.h"
#include <torch/extension.h>
#include <stdio.h>

extern void gcn_aggregate(int m, int nnz, int feat_len,
int *RowPtr, int *ColIdx, float *in_feat, float *out_feat);

extern void gcn_aggregate_balance(int m, int nnz, int feat_len,
int *RowPtr, int *ColIdx, float *in_feat, float *out_feat, warp_info* winfo, int winfo_n);

// call cuSPARSE
extern float gcn_aggregate_cusparse(int m, int nnz, int feat_len,
int *RowPtr, int *ColIdx, float *Values, float *in_feat, float *out_feat, int warmup, int repetitions);

// GCN optimized version
extern void gcn_aggregate_balance(int m, int nnz, int feat_len, int *RowPtr, int *ColIdx, 
float *in_feat, float *out_feat, float *degree, warp_info* winfo, int winfo_n);

extern void gcn_aggregate_scheduled(int m, int nnz, int feat_len,
int *RowPtr, int *ColIdx, float *in_feat, float *out_feat, warp_info** sinfo, int* sinfo_n);

extern void gcn_aggregate_balance_shared(int m, int nnz, int feat_len,
int *RowPtr, int *ColIdx, float *in_feat, float *out_feat, warp_info* winfo, int winfo_n);

extern void gcn_aggregate_fused(int m, int nnz, int feat_len_in, int feat_len_out,
int *RowPtr, int *ColIdx, float *in_feat, float *weight, float *out_feat,
block_info *binfo, int binfo_n);

extern void gcn_aggregate_bin_pack(int m, int nnz, int feat_len,
int *PckPtr, int *PckCont, float *in_feat, float *out_feat, bin_pack_info **bpinfo, int *bpinfo_n);

extern void gcn_aggregate_bin_pack2(int m, int nnz, int feat_len,
int *PckPtr, int *PckCont, float *in_feat, float *out_feat, bin_pack_info *bpinfo, int bpinfo_n);

extern void gcn_aggregate_bin_pack2(int m, int nnz, int feat_len,
int *PckPtr, int *PckCont, float *in_feat, float *out_feat, float *degree,
bin_pack_info *bpinfo, int bpinfo_n);

extern void gcn_aggregate_bin_pack3(int m, int nnz, int feat_len,
int *PckPtr, int *PckCont, int *RowPtr, int *ColIdx, float *in_feat, float *out_feat, 
bin_pack_info *bpinfo, int bpinfo_n, warp_info* spinfo, int spinfo_n);

extern void gin_aggregate_balance(int m, int nnz, int feat_len,
int *RowPtr, int *ColIdx, float *in_feat, float *out_feat, float eps, warp_info* winfo, int winfo_n);

extern void gin_aggregate_bin_pack3(int m, int nnz, int feat_len,
int *PckPtr, int *PckCont, int *RowPtr, int *ColIdx, float *in_feat, float *out_feat, float eps,
bin_pack_info *bpinfo, int bpinfo_n, warp_info* spinfo, int spinfo_n);

extern void gat_aggregate_balance(int m, int nnz, int feat_len, int *RowPtr, int *ColIdx, 
float *in_feat, float *a_vec, float relu_l, float *out_feat, float *sum_vec, float *edge_weight, warp_info* winfo, int winfo_n);

extern void gat_aggregate_balance2(int m, int nnz, int feat_len, int *RowPtr, int *ColIdx,
float *in_feat, float *a_vec, float relu_l, float *out_feat, float *sum_vec, float *edge_weight,
warp_info* winfo, int winfo_n);

extern void gat_aggregate_bin_pack3(int m, int nnz, int feat_len, int *PckPtr, int *PckCont, int *RowPtr, int *ColIdx,
float *in_feat, float *a_vec, float relu_l, float *out_feat, float *max_vec,
bin_pack_info *bpinfo, int bpinfo_n, warp_info* spinfo, int spinfo_n);

extern void sddmm_aggregate_balance(int m, int nnz, int feat_len,
int *RowPtr, int *ColIdx, float *in_feat1, float *in_feat2, float *out_feat, warp_info* winfo, int winfo_n);

extern float sddmm_aggregate_cusparse(int m, int nnz, int feat_len,
int *RowPtr, int *ColIdx, float *Values, float *in_feat1, float *in_feat2, int warmup, int repetitions);

void kg_gcn_run(
    torch::Tensor RowPtr,
    torch::Tensor ColIdx,
    torch::Tensor in_feat,
    torch::Tensor out_feat
)
{
    int m = RowPtr.size(0) - 1;
    int nnz = ColIdx.size(0);
    int feat_len = in_feat.size(1);
    gcn_aggregate(m, nnz, feat_len,
    RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), in_feat.data_ptr<float>(), out_feat.data_ptr<float>());
}

void kg_nn_gcn_fused_run(
    torch::Tensor RowPtr,
    torch::Tensor ColIdx,
    torch::Tensor in_feat,
    torch::Tensor weight,
    torch::Tensor out_feat,
    int64_t ana_add
)
{
    int m = RowPtr.size(0) - 1;
    int nnz = ColIdx.size(0);
    int feat_len_in = in_feat.size(1);
    int feat_len_out = weight.size(1);

    ana_info ana = *((ana_info*)ana_add);
    
    gcn_aggregate_fused(m, nnz, feat_len_in, feat_len_out,
    RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), 
    in_feat.data_ptr<float>(), weight.data_ptr<float>(), out_feat.data_ptr<float>(),
    ana.binfo, ana.binfo_n);
}

void kg_gcn_run_balance(
    torch::Tensor RowPtr,
    torch::Tensor ColIdx,
    torch::Tensor in_feat,
    torch::Tensor out_feat,
    int64_t ana_add
)
{
    int m = RowPtr.size(0) - 1;
    int nnz = ColIdx.size(0);
    int feat_len = in_feat.size(1);

    ana_info ana = *((ana_info*)ana_add);

    if (ana.winfo)
        gcn_aggregate_balance(m, nnz, feat_len, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(),
        in_feat.data_ptr<float>(), out_feat.data_ptr<float>(), (warp_info*)ana.winfo, ana.winfo_n);
    else if (ana.sinfo)
        gcn_aggregate_scheduled(m, nnz, feat_len, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(),
        in_feat.data_ptr<float>(), out_feat.data_ptr<float>(), (warp_info**)ana.sinfo, ana.sinfo_n);
    else if (ana.bp)
    {
        // if (ana.bpinfo)
        //     gcn_aggregate_bin_pack(m, nnz, feat_len, ana.bp->PckPtr, ana.bp->PckCont,
        //     in_feat.data_ptr<float>(), out_feat.data_ptr<float>(), (bin_pack_info**)ana.bpinfo, ana.bpinfo_n);
        
        // if (ana.bpinfo2)
        // {
        // gcn_aggregate_bin_pack2(m, nnz, feat_len, ana.bp->PckPtr, ana.bp->PckCont,
        // in_feat.data_ptr<float>(), out_feat.data_ptr<float>(), (bin_pack_info*)ana.bpinfo2, ana.bpinfo_n2);
        gcn_aggregate_bin_pack3(m, nnz, feat_len, ana.bp->PckPtr, ana.bp->PckCont, 
        ana.bp->RowPtr_sp, ana.bp->ColIdx_sp, in_feat.data_ptr<float>(), out_feat.data_ptr<float>(), 
        (bin_pack_info*)ana.bpinfo2, ana.bpinfo_n2, (warp_info*)ana.spinfo, ana.spinfo_n);
        // }

        // if (ana.spinfo)
        // {
        //     int m_sp = ana.bp->spn;
        //     int nnz_sp = ana.bp->RowPtr_sp_h[m_sp];

        //     gcn_aggregate_balance(m_sp, nnz_sp, feat_len, ana.bp->RowPtr_sp, ana.bp->ColIdx_sp,
        //     in_feat.data_ptr<float>(), out_feat.data_ptr<float>(), (warp_info*)ana.spinfo, ana.spinfo_n);
        // }
    }
    else
        printf("Not implemented!\n");
}

void kg_gcn_run_balance_with_deg(
    torch::Tensor RowPtr,
    torch::Tensor ColIdx,
    torch::Tensor in_feat,
    torch::Tensor out_feat,
    torch::Tensor degree,
    int64_t ana_add
)
{
    int m = RowPtr.size(0) - 1;
    int nnz = ColIdx.size(0);
    int feat_len = in_feat.size(1);

    ana_info ana = *((ana_info*)ana_add);

    if (ana.winfo)
        gcn_aggregate_balance(m, nnz, feat_len, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(),
        in_feat.data_ptr<float>(), out_feat.data_ptr<float>(), degree.data_ptr<float>(), (warp_info*)ana.winfo, ana.winfo_n);
    else if (ana.sinfo)
        gcn_aggregate_scheduled(m, nnz, feat_len, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(),
        in_feat.data_ptr<float>(), out_feat.data_ptr<float>(), (warp_info**)ana.sinfo, ana.sinfo_n);
    else if (ana.bp)
    {
        if (ana.bpinfo)
            gcn_aggregate_bin_pack(m, nnz, feat_len, ana.bp->PckPtr, ana.bp->PckCont,
            in_feat.data_ptr<float>(), out_feat.data_ptr<float>(), (bin_pack_info**)ana.bpinfo, ana.bpinfo_n);
        
        if (ana.bpinfo2)
            gcn_aggregate_bin_pack2(m, nnz, feat_len, ana.bp->PckPtr, ana.bp->PckCont,
            in_feat.data_ptr<float>(), out_feat.data_ptr<float>(), degree.data_ptr<float>(), (bin_pack_info*)ana.bpinfo2, ana.bpinfo_n2);

        if (ana.spinfo)
        {
            int m_sp = ana.bp->spn;
            int nnz_sp = ana.bp->RowPtr_sp_h[m_sp];

            gcn_aggregate_balance(m_sp, nnz_sp, feat_len, ana.bp->RowPtr_sp, ana.bp->ColIdx_sp,
            in_feat.data_ptr<float>(), out_feat.data_ptr<float>(), degree.data_ptr<float>(), (warp_info*)ana.spinfo, ana.spinfo_n);
        }
    }
    else
        printf("Not implemented!\n");
}

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
)
{
    int m = RowPtr.size(0) - 1;
    int nnz = ColIdx.size(0);
    int feat_len = in_feat.size(1);

    ana_info ana = *((ana_info*)ana_add);

    //void gat_aggregate_balance(int m, int nnz, int feat_len, int *RowPtr, int *ColIdx, 
    //float *in_feat, float *a_vec, float relu_l, float *out_feat, float *max_vec, warp_info* winfo, int winfo_n)
    torch::Tensor a_vec = torch::mm(in_feat, a_weight);

    if (ana.winfo)
        gat_aggregate_balance(m, nnz, feat_len, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(),
        in_feat.data_ptr<float>(), a_vec.data_ptr<float>(), relu_l, out_feat.data_ptr<float>(), 
        sum_vec.data_ptr<float>(), edge_weight.data_ptr<float>(), (warp_info*)ana.winfo, ana.winfo_n);
    else if (ana.bp)
    {
        // if (ana.bpinfo)
        //     gcn_aggregate_bin_pack(m, nnz, feat_len, ana.bp->PckPtr, ana.bp->PckCont,
        //     in_feat.data_ptr<float>(), out_feat.data_ptr<float>(), (bin_pack_info**)ana.bpinfo, ana.bpinfo_n);
        
        // if (ana.bpinfo2)
        // {
        // gcn_aggregate_bin_pack2(m, nnz, feat_len, ana.bp->PckPtr, ana.bp->PckCont,
        // in_feat.data_ptr<float>(), out_feat.data_ptr<float>(), (bin_pack_info*)ana.bpinfo2, ana.bpinfo_n2);
        // gat_aggregate_bin_pack3(m, nnz, feat_len, ana.bp->PckPtr, ana.bp->PckCont, ana.bp->RowPtr_sp, ana.bp->ColIdx_sp, 
        // in_feat.data_ptr<float>(), a_vec.data_ptr<float>(), relu_l, out_feat.data_ptr<float>(), max_vec.data_ptr<float>(), 
        // (bin_pack_info*)ana.bpinfo2, ana.bpinfo_n2, (warp_info*)ana.spinfo, ana.spinfo_n);
        // }

        // if (ana.spinfo)
        // {
        //     int m_sp = ana.bp->spn;
        //     int nnz_sp = ana.bp->RowPtr_sp_h[m_sp];

        //     gcn_aggregate_balance(m_sp, nnz_sp, feat_len, ana.bp->RowPtr_sp, ana.bp->ColIdx_sp,
        //     in_feat.data_ptr<float>(), out_feat.data_ptr<float>(), (warp_info*)ana.spinfo, ana.spinfo_n);
        // }
        printf("Not implemented!\n");
        return;
    }
    else
    {
        printf("Not implemented!\n");
        return;
    }

}

void kg_gin_run_balance(
    torch::Tensor RowPtr,
    torch::Tensor ColIdx,
    torch::Tensor in_feat,
    torch::Tensor out_feat,
    float eps,
    int64_t ana_add
)
{
    int m = RowPtr.size(0) - 1;
    int nnz = ColIdx.size(0);
    int feat_len = in_feat.size(1);

    ana_info ana = *((ana_info*)ana_add);

    if (ana.winfo)
        gin_aggregate_balance(m, nnz, feat_len, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(),
        in_feat.data_ptr<float>(), out_feat.data_ptr<float>(), eps, (warp_info*)ana.winfo, ana.winfo_n);
    else if (ana.bp)
    {
        gin_aggregate_bin_pack3(m, nnz, feat_len, ana.bp->PckPtr, ana.bp->PckCont, 
        ana.bp->RowPtr_sp, ana.bp->ColIdx_sp, in_feat.data_ptr<float>(), out_feat.data_ptr<float>(), eps,
        (bin_pack_info*)ana.bpinfo2, ana.bpinfo_n2, (warp_info*)ana.spinfo, ana.spinfo_n);
    }
    else
        printf("Not implemented!\n");
}

void kg_gcn_run_balance_shared(
    torch::Tensor RowPtr,
    torch::Tensor ColIdx,
    torch::Tensor in_feat,
    torch::Tensor out_feat,
    int64_t ana_add
)
{
    int m = RowPtr.size(0) - 1;
    int nnz = ColIdx.size(0);
    int feat_len = in_feat.size(1);

    ana_info ana = *((ana_info*)ana_add);

    gcn_aggregate_balance_shared(m, nnz, feat_len, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(),
    in_feat.data_ptr<float>(), out_feat.data_ptr<float>(), (warp_info*)ana.winfo, ana.winfo_n);
}

float kg_gcn_run_cusparse(
    torch::Tensor RowPtr,
    torch::Tensor ColIdx,
    torch::Tensor Values,
    torch::Tensor in_feat,
    torch::Tensor out_feat,
    int warmup,
    int repetitions
)
{
    int m = RowPtr.size(0) - 1;
    int nnz = ColIdx.size(0);
    int feat_len = in_feat.size(1);

    // float kernel_time = 0;
    float kernel_time = gcn_aggregate_cusparse(m, nnz, feat_len, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), Values.data_ptr<float>(),
    in_feat.data_ptr<float>(), out_feat.data_ptr<float>(), warmup, repetitions);

    return kernel_time;
}

void kg_sddmm_run_balance(
    torch::Tensor RowPtr,
    torch::Tensor ColIdx,
    torch::Tensor in_feat1,
    torch::Tensor in_feat2,
    torch::Tensor out_feat,
    int64_t ana_add
)
{
    int m = RowPtr.size(0) - 1;
    int nnz = ColIdx.size(0);
    int feat_len = in_feat1.size(1);

    ana_info ana = *((ana_info*)ana_add);

    if (ana.winfo)
        sddmm_aggregate_balance(m, nnz, feat_len, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(),
        in_feat1.data_ptr<float>(), in_feat2.data_ptr<float>(),
        out_feat.data_ptr<float>(), (warp_info*)ana.winfo, ana.winfo_n);
}

// transpose(in_feat1) * transpose(in_feat2)
float kg_sddmm_run_cusparse(
    torch::Tensor RowPtr,
    torch::Tensor ColIdx,
    torch::Tensor in_feat1,
    torch::Tensor in_feat2,
    int feat_len,
    torch::Tensor out_feat,
    int warmup,
    int repetitions
)
{
    int m = RowPtr.size(0) - 1;
    int nnz = ColIdx.size(0);
    //int feat_len = in_feat1.size(0);

    // float kernel_time = 0;
    float kernel_time = sddmm_aggregate_cusparse(m, nnz, feat_len, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(),
    out_feat.data_ptr<float>(), in_feat1.data_ptr<float>(), in_feat2.data_ptr<float>(), warmup, repetitions);

    return kernel_time;
}