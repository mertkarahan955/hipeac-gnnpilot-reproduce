#include <torch/extension.h>
#include <cub/cub.cuh>
#include "preprocessing.h"
#define WARP_SIZE 32
#define N_BLOCK_SIZE 128
#define NE_BLOCK_SIZE 128
#define NE_BLOCK_WARP_SIZE (NE_BLOCK_SIZE / WARP_SIZE)
#define E_BLOCK_SIZE 128
#define E_BLOCK_WARP_SIZE (E_BLOCK_SIZE / WARP_SIZE)
#define HE_BLOCK_SIZE 128
#define HE_BLOCK_WARP_SIZE (HE_BLOCK_SIZE / WARP_SIZE)
#define NGD_BLOCK_SIZE 128
#define NGD_BLOCK_WARP_SIZE (NGD_BLOCK_SIZE / WARP_SIZE)
#define DP_NNZ_PER_WARP 128
#define DP_BLOCK_SIZE 128
#define DP_NNZ_PER_BLOCK (DP_NNZ_PER_WARP * DP_BLOCK_SIZE / WARP_SIZE)
#define kg_max(a, b) ((a)>(b)? (a): (b))
#define kg_min(a, b) ((a)<(b)? (a): (b))

extern int64_t preprocessing_cuda(int m, int nnz, int *RowPtr, int *ColIdx, bool long_dynamic);
int64_t preprocessing(torch::Tensor RowPtr, torch::Tensor ColIdx, int64_t long_dynamic) {
fprintf(stderr, "DEBUG preprocessing (wrapper): RowPtr.size(0)=%ld, ColIdx.size(0)=%ld\n", 
        RowPtr.size(0), ColIdx.size(0));
fflush(stderr);
int m = RowPtr.size(0) - 1;
int nnz = ColIdx.size(0);
fprintf(stderr, "DEBUG preprocessing (wrapper): m=%d, nnz=%d, calling preprocessing_cuda...\n", m, nnz);
fflush(stderr);
int64_t result = preprocessing_cuda(m, nnz, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), long_dynamic);
fprintf(stderr, "DEBUG preprocessing (wrapper): preprocessing_cuda returned %lld\n", (long long)result);
fflush(stderr);
return result;
}

__device__ static float atomicMax_float(float* addr, float val) {
int* addr_as_int = (int*)addr;
int old = *addr_as_int;
int expected;
do {
expected = old;
old = ::atomicCAS(addr_as_int, expected,
__float_as_int(::fmaxf(val, __int_as_float(expected))));
} while (expected != old);
return __int_as_float(old);
}

__global__ void k_0_fop_0(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* e, float* h) {
int lane_id = threadIdx.x % WARP_SIZE;
int row_tmp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
if (row_tmp >= numnodes) return;
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];
}
}
__global__ void k_0_fop_1(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* h, float* lr, float* em) {
int lane_id = threadIdx.x % WARP_SIZE;
int row_tmp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
if (row_tmp >= numnodes) return;
float res_tmp = -99999;
for (int nnz_tmp = RowPtr[row_tmp]+lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) em[row_tmp * 1 + 0] = res_tmp;
}
__global__ void k_0_fop_2(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* h, float* em, float* ho) {
int lane_id = threadIdx.x % WARP_SIZE;
int row_tmp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
if (row_tmp >= numnodes) return;
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);
}
}
__global__ void k_0_fop_3(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* ho, float* es) {
int lane_id = threadIdx.x % WARP_SIZE;
int row_tmp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
if (row_tmp >= numnodes) return;
float res_tmp = 0;
for (int nnz_tmp = RowPtr[row_tmp]+lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) es[row_tmp * 1 + 0] = res_tmp;
}
__global__ void k_0_fop_4(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* ho, float* f, float* es, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
int row_tmp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
if (row_tmp >= numnodes) return;
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = RowPtr[row_tmp]; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
fo[row_tmp * featlen + k_tmp] = res_tmp;
}
}
void gat_kernel_0(int64_t info_addr, torch::Tensor RowPtr, torch::Tensor ColIdx, int64_t featlen, torch::Tensor f, torch::Tensor fo, torch::Tensor we, torch::Tensor lr, torch::Tensor e, torch::Tensor em, torch::Tensor es, torch::Tensor h, torch::Tensor ho) {
kg_info* info_ = (kg_info*)info_addr;
if (info_ == nullptr) {
  throw std::runtime_error("info_ is null in gat_kernel_0");
}
if (info_->rp_info == nullptr || info_->rp_n == nullptr || info_->ep_info == nullptr || info_->ep_n == nullptr || info_->ng_info == nullptr || info_->ng_n == nullptr) {
  throw std::runtime_error("Some preprocessing pointers are null in gat_kernel_0");
}
int numnodes = RowPtr.size(0) - 1;
int numedges = ColIdx.size(0);
int num_n_block = (numnodes + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
int num_ne_block = (numnodes + NE_BLOCK_WARP_SIZE - 1) / NE_BLOCK_WARP_SIZE;
int num_e_block = info_->ep_n_host;
int num_he_block = info_->rp_n_host;
int num_ngd_block = (info_->ng_n_host + NGD_BLOCK_WARP_SIZE - 1) / NGD_BLOCK_WARP_SIZE;
k_0_fop_0<<<num_ne_block, NE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, e.data_ptr<float>(), h.data_ptr<float>());
k_0_fop_1<<<num_ne_block, NE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, h.data_ptr<float>(), lr.data_ptr<float>(), em.data_ptr<float>());
k_0_fop_2<<<num_ne_block, NE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, h.data_ptr<float>(), em.data_ptr<float>(), ho.data_ptr<float>());
k_0_fop_3<<<num_ne_block, NE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, ho.data_ptr<float>(), es.data_ptr<float>());
k_0_fop_4<<<num_ne_block, NE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, ho.data_ptr<float>(), f.data_ptr<float>(), es.data_ptr<float>(), fo.data_ptr<float>());
}
__global__ void k_1_fop_0(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* e, float* h, float* lr, float* em, float* ho, float* es) {
int lane_id = threadIdx.x % WARP_SIZE;
int row_tmp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
if (row_tmp >= numnodes) return;
{ for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];
}
}
{ float res_tmp = -99999;
for (int nnz_tmp = RowPtr[row_tmp]+lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) em[row_tmp * 1 + 0] = res_tmp;
}
{ for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);
}
}
{ float res_tmp = 0;
for (int nnz_tmp = RowPtr[row_tmp]+lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) es[row_tmp * 1 + 0] = res_tmp;
}
}
__global__ void k_1_fop_1(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* ho, float* f, float* es, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
int row_tmp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
if (row_tmp >= numnodes) return;
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = RowPtr[row_tmp]; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
fo[row_tmp * featlen + k_tmp] = res_tmp;
}
}
void gat_kernel_1(int64_t info_addr, torch::Tensor RowPtr, torch::Tensor ColIdx, int64_t featlen, torch::Tensor f, torch::Tensor fo, torch::Tensor we, torch::Tensor lr, torch::Tensor e, torch::Tensor em, torch::Tensor es, torch::Tensor h, torch::Tensor ho) {
kg_info* info_ = (kg_info*)info_addr;
if (info_ == nullptr) {
  throw std::runtime_error("info_ is null in gat_kernel_1");
}
if (info_->rp_info == nullptr || info_->rp_n == nullptr || info_->ep_info == nullptr || info_->ep_n == nullptr || info_->ng_info == nullptr || info_->ng_n == nullptr) {
  throw std::runtime_error("Some preprocessing pointers are null in gat_kernel_1");
}
int numnodes = RowPtr.size(0) - 1;
int numedges = ColIdx.size(0);
int num_n_block = (numnodes + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
int num_ne_block = (numnodes + NE_BLOCK_WARP_SIZE - 1) / NE_BLOCK_WARP_SIZE;
int num_e_block = info_->ep_n_host;
int num_he_block = info_->rp_n_host;
int num_ngd_block = (info_->ng_n_host + NGD_BLOCK_WARP_SIZE - 1) / NGD_BLOCK_WARP_SIZE;
k_1_fop_0<<<num_ne_block, NE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, e.data_ptr<float>(), h.data_ptr<float>(), lr.data_ptr<float>(), em.data_ptr<float>(), ho.data_ptr<float>(), es.data_ptr<float>());
k_1_fop_1<<<num_ne_block, NE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, ho.data_ptr<float>(), f.data_ptr<float>(), es.data_ptr<float>(), fo.data_ptr<float>());
}
__global__ void k2_fop0_dp4(int *RowPtr, int *ColIdx, int row_tmp, int featlen, float* ho, float* f, float* es, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
int idx_st = RowPtr[row_tmp] + blockIdx.x * DP_NNZ_PER_BLOCK + threadIdx.x / WARP_SIZE * DP_NNZ_PER_WARP;
int idx_ed = idx_st + DP_NNZ_PER_WARP;
if (idx_ed > RowPtr[row_tmp + 1]) idx_ed = RowPtr[row_tmp + 1];
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = idx_st; nnz_tmp < idx_ed; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];
}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
atomicAdd(&fo[row_tmp * featlen + k_tmp], res_tmp);
}
}
__global__ void k_2_fop_0(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* e, float* h, float* lr, float* em, float* ho, float* es, float* f, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *info_n) return;
row_panel info_tmp = info_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
int row_tmp_ne_st = info_tmp.row_st + local_wid;
int row_tmp_ne_ed = info_tmp.row_ed;
if (info_tmp.col_st == -1) {
{ for (int row_tmp = row_tmp_ne_st; row_tmp < row_tmp_ne_ed; row_tmp += HE_BLOCK_WARP_SIZE) {
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];
}
}
}
__syncthreads();
{ for (int row_tmp = row_tmp_ne_st; row_tmp < row_tmp_ne_ed; row_tmp += HE_BLOCK_WARP_SIZE) {
float res_tmp = -99999;
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;
}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) em[row_tmp * 1 + 0] = res_tmp;
}
}
__syncthreads();
{ for (int row_tmp = row_tmp_ne_st; row_tmp < row_tmp_ne_ed; row_tmp += HE_BLOCK_WARP_SIZE) {
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);
}
}
}
__syncthreads();
{ for (int row_tmp = row_tmp_ne_st; row_tmp < row_tmp_ne_ed; row_tmp += HE_BLOCK_WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];
}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) es[row_tmp * 1 + 0] = res_tmp;
}
}
__syncthreads();
{ for (int row_tmp = row_tmp_ne_st; row_tmp < row_tmp_ne_ed; row_tmp += HE_BLOCK_WARP_SIZE) {
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = RowPtr[row_tmp]; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];
}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
fo[row_tmp * featlen + k_tmp] = res_tmp;
}
}
}
__syncthreads();
}
else {
int row_tmp = info_tmp.row_st;
int nnzs = RowPtr[row_tmp + 1] - RowPtr[row_tmp];
int dynamic_blocks = (nnzs + DP_NNZ_PER_BLOCK - 1) / DP_NNZ_PER_BLOCK;
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];}__syncthreads();
{
float res_tmp = -99999;
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) {
atomicMax_float(&em[row_tmp * 1 + 0], res_tmp);
}
}__syncthreads();
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);}__syncthreads();
{
float res_tmp = 0;
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) {
atomicAdd(&es[row_tmp * 1 + 0], res_tmp);
}
}__syncthreads();
if (threadIdx.x == 0) k2_fop0_dp4<<<dynamic_blocks, DP_BLOCK_SIZE>>>(RowPtr, ColIdx, row_tmp, featlen, ho, f, es, fo);
}
}
void gat_kernel_2(int64_t info_addr, torch::Tensor RowPtr, torch::Tensor ColIdx, int64_t featlen, torch::Tensor f, torch::Tensor fo, torch::Tensor we, torch::Tensor lr, torch::Tensor e, torch::Tensor em, torch::Tensor es, torch::Tensor h, torch::Tensor ho) {
kg_info* info_ = (kg_info*)info_addr;
if (info_ == nullptr) {
  throw std::runtime_error("info_ is null in gat_kernel_2");
}
if (info_->rp_info == nullptr || info_->rp_n == nullptr || info_->ep_info == nullptr || info_->ep_n == nullptr || info_->ng_info == nullptr || info_->ng_n == nullptr) {
  throw std::runtime_error("Some preprocessing pointers are null in gat_kernel_2");
}
int numnodes = RowPtr.size(0) - 1;
int numedges = ColIdx.size(0);
int num_n_block = (numnodes + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
int num_ne_block = (numnodes + NE_BLOCK_WARP_SIZE - 1) / NE_BLOCK_WARP_SIZE;
int num_e_block = info_->ep_n_host;
int num_he_block = info_->rp_n_host;
int num_ngd_block = (info_->ng_n_host + NGD_BLOCK_WARP_SIZE - 1) / NGD_BLOCK_WARP_SIZE;
k_2_fop_0<<<num_he_block, HE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, e.data_ptr<float>(), h.data_ptr<float>(), lr.data_ptr<float>(), em.data_ptr<float>(), ho.data_ptr<float>(), es.data_ptr<float>(), f.data_ptr<float>(), fo.data_ptr<float>());
}
__global__ void k_3_fop_0(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* e, float* h) {
int lane_id = threadIdx.x % WARP_SIZE;
int row_tmp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
if (row_tmp >= numnodes) return;
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];
}
}
__global__ void k_3_fop_1(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* h, float* lr, float* em) {
int lane_id = threadIdx.x % WARP_SIZE;
int row_tmp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
if (row_tmp >= numnodes) return;
float res_tmp = -99999;
for (int nnz_tmp = RowPtr[row_tmp]+lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) em[row_tmp * 1 + 0] = res_tmp;
}
__global__ void k_3_fop_2(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* h, float* em, float* ho) {
int lane_id = threadIdx.x % WARP_SIZE;
int row_tmp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
if (row_tmp >= numnodes) return;
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);
}
}
__global__ void k_3_fop_3(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* ho, float* es) {
int lane_id = threadIdx.x % WARP_SIZE;
int row_tmp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
if (row_tmp >= numnodes) return;
float res_tmp = 0;
for (int nnz_tmp = RowPtr[row_tmp]+lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) es[row_tmp * 1 + 0] = res_tmp;
}
__global__ void k_3_fop_4(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* ho, float* f, float* es, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *ep_n) return;
row_panel info_tmp = ep_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (info_tmp.col_st == -1){ 
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
int nnz_per_warp = (nnzs + E_BLOCK_WARP_SIZE - 1) / E_BLOCK_WARP_SIZE;
int warp_col_st = col_st + nnz_per_warp * local_wid;
int warp_col_ed = (warp_col_st + nnz_per_warp < col_ed)? warp_col_st + nnz_per_warp: col_ed;
int warp_row_st = info_tmp.row_st, warp_row_ed;
while (warp_row_st < info_tmp.row_ed - 1 && warp_col_st > RowPtr[warp_row_st + 1]) {
warp_row_st++;
}
warp_row_ed = warp_row_st;
while (warp_row_ed < info_tmp.row_ed && warp_col_ed > RowPtr[warp_row_ed]) {
warp_row_ed++;
}
if (warp_col_ed <= col_ed) {
for (int row_tmp = warp_row_st; row_tmp < warp_row_ed; row_tmp++) {
int current_col_st = RowPtr[row_tmp];
int current_col_ed = RowPtr[row_tmp + 1];
if (row_tmp == warp_row_st) current_col_st = warp_col_st;
if (row_tmp == warp_row_ed - 1) current_col_ed = warp_col_ed;
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = current_col_st; nnz_tmp < current_col_ed; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
atomicAdd(&fo[row_tmp * featlen + k_tmp], res_tmp);
}}}
}
else {
int row_tmp = info_tmp.row_st;
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = info_tmp.col_st + local_wid; nnz_tmp < info_tmp.col_ed; nnz_tmp+=HE_BLOCK_WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];
}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
atomicAdd(&fo[row_tmp * featlen + k_tmp], res_tmp);
}
}
}
void gat_kernel_3(int64_t info_addr, torch::Tensor RowPtr, torch::Tensor ColIdx, int64_t featlen, torch::Tensor f, torch::Tensor fo, torch::Tensor we, torch::Tensor lr, torch::Tensor e, torch::Tensor em, torch::Tensor es, torch::Tensor h, torch::Tensor ho) {
kg_info* info_ = (kg_info*)info_addr;
if (info_ == nullptr) {
  throw std::runtime_error("info_ is null in gat_kernel_3");
}
if (info_->rp_info == nullptr || info_->rp_n == nullptr || info_->ep_info == nullptr || info_->ep_n == nullptr || info_->ng_info == nullptr || info_->ng_n == nullptr) {
  throw std::runtime_error("Some preprocessing pointers are null in gat_kernel_3");
}
int numnodes = RowPtr.size(0) - 1;
int numedges = ColIdx.size(0);
int num_n_block = (numnodes + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
int num_ne_block = (numnodes + NE_BLOCK_WARP_SIZE - 1) / NE_BLOCK_WARP_SIZE;
int num_e_block = info_->ep_n_host;
int num_he_block = info_->rp_n_host;
int num_ngd_block = (info_->ng_n_host + NGD_BLOCK_WARP_SIZE - 1) / NGD_BLOCK_WARP_SIZE;
k_3_fop_0<<<num_ne_block, NE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, e.data_ptr<float>(), h.data_ptr<float>());
k_3_fop_1<<<num_ne_block, NE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, h.data_ptr<float>(), lr.data_ptr<float>(), em.data_ptr<float>());
k_3_fop_2<<<num_ne_block, NE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, h.data_ptr<float>(), em.data_ptr<float>(), ho.data_ptr<float>());
k_3_fop_3<<<num_ne_block, NE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, ho.data_ptr<float>(), es.data_ptr<float>());
k_3_fop_4<<<num_e_block, E_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, ho.data_ptr<float>(), f.data_ptr<float>(), es.data_ptr<float>(), fo.data_ptr<float>());
}
__global__ void k_4_fop_0(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* e, float* h, float* lr, float* em, float* ho, float* es) {
int lane_id = threadIdx.x % WARP_SIZE;
int row_tmp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
if (row_tmp >= numnodes) return;
{ for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];
}
}
{ float res_tmp = -99999;
for (int nnz_tmp = RowPtr[row_tmp]+lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) em[row_tmp * 1 + 0] = res_tmp;
}
{ for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);
}
}
{ float res_tmp = 0;
for (int nnz_tmp = RowPtr[row_tmp]+lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) es[row_tmp * 1 + 0] = res_tmp;
}
}
__global__ void k_4_fop_1(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* ho, float* f, float* es, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *ep_n) return;
row_panel info_tmp = ep_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (info_tmp.col_st == -1){ 
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
int nnz_per_warp = (nnzs + E_BLOCK_WARP_SIZE - 1) / E_BLOCK_WARP_SIZE;
int warp_col_st = col_st + nnz_per_warp * local_wid;
int warp_col_ed = (warp_col_st + nnz_per_warp < col_ed)? warp_col_st + nnz_per_warp: col_ed;
int warp_row_st = info_tmp.row_st, warp_row_ed;
while (warp_row_st < info_tmp.row_ed - 1 && warp_col_st > RowPtr[warp_row_st + 1]) {
warp_row_st++;
}
warp_row_ed = warp_row_st;
while (warp_row_ed < info_tmp.row_ed && warp_col_ed > RowPtr[warp_row_ed]) {
warp_row_ed++;
}
if (warp_col_ed <= col_ed) {
for (int row_tmp = warp_row_st; row_tmp < warp_row_ed; row_tmp++) {
int current_col_st = RowPtr[row_tmp];
int current_col_ed = RowPtr[row_tmp + 1];
if (row_tmp == warp_row_st) current_col_st = warp_col_st;
if (row_tmp == warp_row_ed - 1) current_col_ed = warp_col_ed;
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = current_col_st; nnz_tmp < current_col_ed; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
atomicAdd(&fo[row_tmp * featlen + k_tmp], res_tmp);
}}}
}
else {
int row_tmp = info_tmp.row_st;
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = info_tmp.col_st + local_wid; nnz_tmp < info_tmp.col_ed; nnz_tmp+=HE_BLOCK_WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];
}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
atomicAdd(&fo[row_tmp * featlen + k_tmp], res_tmp);
}
}
}
void gat_kernel_4(int64_t info_addr, torch::Tensor RowPtr, torch::Tensor ColIdx, int64_t featlen, torch::Tensor f, torch::Tensor fo, torch::Tensor we, torch::Tensor lr, torch::Tensor e, torch::Tensor em, torch::Tensor es, torch::Tensor h, torch::Tensor ho) {
kg_info* info_ = (kg_info*)info_addr;
if (info_ == nullptr) {
  throw std::runtime_error("info_ is null in gat_kernel_4");
}
if (info_->rp_info == nullptr || info_->rp_n == nullptr || info_->ep_info == nullptr || info_->ep_n == nullptr || info_->ng_info == nullptr || info_->ng_n == nullptr) {
  throw std::runtime_error("Some preprocessing pointers are null in gat_kernel_4");
}
int numnodes = RowPtr.size(0) - 1;
int numedges = ColIdx.size(0);
int num_n_block = (numnodes + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
int num_ne_block = (numnodes + NE_BLOCK_WARP_SIZE - 1) / NE_BLOCK_WARP_SIZE;
int num_e_block = info_->ep_n_host;
int num_he_block = info_->rp_n_host;
int num_ngd_block = (info_->ng_n_host + NGD_BLOCK_WARP_SIZE - 1) / NGD_BLOCK_WARP_SIZE;
k_4_fop_0<<<num_ne_block, NE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, e.data_ptr<float>(), h.data_ptr<float>(), lr.data_ptr<float>(), em.data_ptr<float>(), ho.data_ptr<float>(), es.data_ptr<float>());
k_4_fop_1<<<num_e_block, E_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, ho.data_ptr<float>(), f.data_ptr<float>(), es.data_ptr<float>(), fo.data_ptr<float>());
}
__global__ void k5_fop0_dp4(int *RowPtr, int *ColIdx, int row_tmp, int featlen, float* ho, float* f, float* es, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
int idx_st = RowPtr[row_tmp] + blockIdx.x * DP_NNZ_PER_BLOCK + threadIdx.x / WARP_SIZE * DP_NNZ_PER_WARP;
int idx_ed = idx_st + DP_NNZ_PER_WARP;
if (idx_ed > RowPtr[row_tmp + 1]) idx_ed = RowPtr[row_tmp + 1];
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = idx_st; nnz_tmp < idx_ed; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];
}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
atomicAdd(&fo[row_tmp * featlen + k_tmp], res_tmp);
}
}
__global__ void k_5_fop_0(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* e, float* h, float* lr, float* em, float* ho, float* es, float* f, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *info_n) return;
row_panel info_tmp = info_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
int row_tmp_ne_st = info_tmp.row_st + local_wid;
int row_tmp_ne_ed = info_tmp.row_ed;
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
int nnz_per_warp = (nnzs + HE_BLOCK_WARP_SIZE - 1) / HE_BLOCK_WARP_SIZE;
int warp_col_st = col_st + nnz_per_warp * local_wid;
int warp_col_ed = (warp_col_st + nnz_per_warp < col_ed)? warp_col_st + nnz_per_warp: col_ed;
int warp_row_st = info_tmp.row_st, warp_row_ed;
while (warp_row_st < info_tmp.row_ed - 1 && warp_col_st > RowPtr[warp_row_st + 1]) {
warp_row_st++;
}
warp_row_ed = warp_row_st;
while (warp_row_ed < info_tmp.row_ed && warp_col_ed > RowPtr[warp_row_ed]) {
warp_row_ed++;
}
if (info_tmp.col_st == -1) {
{ for (int row_tmp = row_tmp_ne_st; row_tmp < row_tmp_ne_ed; row_tmp += HE_BLOCK_WARP_SIZE) {
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];
}
}
}
__syncthreads();
{ for (int row_tmp = row_tmp_ne_st; row_tmp < row_tmp_ne_ed; row_tmp += HE_BLOCK_WARP_SIZE) {
float res_tmp = -99999;
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;
}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) em[row_tmp * 1 + 0] = res_tmp;
}
}
__syncthreads();
{ for (int row_tmp = row_tmp_ne_st; row_tmp < row_tmp_ne_ed; row_tmp += HE_BLOCK_WARP_SIZE) {
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);
}
}
}
__syncthreads();
{ for (int row_tmp = row_tmp_ne_st; row_tmp < row_tmp_ne_ed; row_tmp += HE_BLOCK_WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];
}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) es[row_tmp * 1 + 0] = res_tmp;
}
}
__syncthreads();
{ if (warp_col_ed <= col_ed) {
for (int row_tmp = warp_row_st; row_tmp < warp_row_ed; row_tmp++) {
int current_col_st = RowPtr[row_tmp];
int current_col_ed = RowPtr[row_tmp + 1];
if (row_tmp == warp_row_st) current_col_st = warp_col_st;
if (row_tmp == warp_row_ed - 1) current_col_ed = warp_col_ed;
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = current_col_st; nnz_tmp < current_col_ed; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];
}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
atomicAdd(&fo[row_tmp * featlen + k_tmp], res_tmp);
}}}
}
__syncthreads();
}
else {
int row_tmp = info_tmp.row_st;
int nnzs = RowPtr[row_tmp + 1] - RowPtr[row_tmp];
int dynamic_blocks = (nnzs + DP_NNZ_PER_BLOCK - 1) / DP_NNZ_PER_BLOCK;
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];}__syncthreads();
{
float res_tmp = -99999;
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) {
atomicMax_float(&em[row_tmp * 1 + 0], res_tmp);
}
}__syncthreads();
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);}__syncthreads();
{
float res_tmp = 0;
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) {
atomicAdd(&es[row_tmp * 1 + 0], res_tmp);
}
}__syncthreads();
if (threadIdx.x == 0) k5_fop0_dp4<<<dynamic_blocks, DP_BLOCK_SIZE>>>(RowPtr, ColIdx, row_tmp, featlen, ho, f, es, fo);
}
}
void gat_kernel_5(int64_t info_addr, torch::Tensor RowPtr, torch::Tensor ColIdx, int64_t featlen, torch::Tensor f, torch::Tensor fo, torch::Tensor we, torch::Tensor lr, torch::Tensor e, torch::Tensor em, torch::Tensor es, torch::Tensor h, torch::Tensor ho) {
kg_info* info_ = (kg_info*)info_addr;
if (info_ == nullptr) {
  throw std::runtime_error("info_ is null in gat_kernel_5");
}
if (info_->rp_info == nullptr || info_->rp_n == nullptr || info_->ep_info == nullptr || info_->ep_n == nullptr || info_->ng_info == nullptr || info_->ng_n == nullptr) {
  throw std::runtime_error("Some preprocessing pointers are null in gat_kernel_5");
}
int numnodes = RowPtr.size(0) - 1;
int numedges = ColIdx.size(0);
int num_n_block = (numnodes + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
int num_ne_block = (numnodes + NE_BLOCK_WARP_SIZE - 1) / NE_BLOCK_WARP_SIZE;
int num_e_block = info_->ep_n_host;
int num_he_block = info_->rp_n_host;
int num_ngd_block = (info_->ng_n_host + NGD_BLOCK_WARP_SIZE - 1) / NGD_BLOCK_WARP_SIZE;
k_5_fop_0<<<num_he_block, HE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, e.data_ptr<float>(), h.data_ptr<float>(), lr.data_ptr<float>(), em.data_ptr<float>(), ho.data_ptr<float>(), es.data_ptr<float>(), f.data_ptr<float>(), fo.data_ptr<float>());
}
__global__ void k_6_fop_0(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* e, float* h) {
int lane_id = threadIdx.x % WARP_SIZE;
int row_tmp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
if (row_tmp >= numnodes) return;
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];
}
}
__global__ void k_6_fop_1(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* h, float* lr, float* em) {
int lane_id = threadIdx.x % WARP_SIZE;
int row_tmp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
if (row_tmp >= numnodes) return;
float res_tmp = -99999;
for (int nnz_tmp = RowPtr[row_tmp]+lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) em[row_tmp * 1 + 0] = res_tmp;
}
__global__ void k_6_fop_2(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* h, float* em, float* ho) {
int lane_id = threadIdx.x % WARP_SIZE;
int row_tmp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
if (row_tmp >= numnodes) return;
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);
}
}
__global__ void k_6_fop_3(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* ho, float* es) {
int lane_id = threadIdx.x % WARP_SIZE;
int row_tmp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
if (row_tmp >= numnodes) return;
float res_tmp = 0;
for (int nnz_tmp = RowPtr[row_tmp]+lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) es[row_tmp * 1 + 0] = res_tmp;
}
__global__ void k_6_fop_4(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* ho, float* f, float* es, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (global_wid >= *ng_n) return;
neighbor_group ng_tmp = ng_list[global_wid];
int row_tmp = ng_tmp.row_st;
int col_st = ng_tmp.col_st;
int col_ed = ng_tmp.col_st + NG_SIZE;
if (col_ed >= RowPtr[row_tmp + 1]) col_ed = RowPtr[row_tmp + 1];
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = col_st; nnz_tmp < col_ed; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
atomicAdd(&fo[row_tmp * featlen + k_tmp], res_tmp);
}
}
void gat_kernel_6(int64_t info_addr, torch::Tensor RowPtr, torch::Tensor ColIdx, int64_t featlen, torch::Tensor f, torch::Tensor fo, torch::Tensor we, torch::Tensor lr, torch::Tensor e, torch::Tensor em, torch::Tensor es, torch::Tensor h, torch::Tensor ho) {
kg_info* info_ = (kg_info*)info_addr;
if (info_ == nullptr) {
  throw std::runtime_error("info_ is null in gat_kernel_6");
}
if (info_->rp_info == nullptr || info_->rp_n == nullptr || info_->ep_info == nullptr || info_->ep_n == nullptr || info_->ng_info == nullptr || info_->ng_n == nullptr) {
  throw std::runtime_error("Some preprocessing pointers are null in gat_kernel_6");
}
int numnodes = RowPtr.size(0) - 1;
int numedges = ColIdx.size(0);
int num_n_block = (numnodes + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
int num_ne_block = (numnodes + NE_BLOCK_WARP_SIZE - 1) / NE_BLOCK_WARP_SIZE;
int num_e_block = info_->ep_n_host;
int num_he_block = info_->rp_n_host;
int num_ngd_block = (info_->ng_n_host + NGD_BLOCK_WARP_SIZE - 1) / NGD_BLOCK_WARP_SIZE;
k_6_fop_0<<<num_ne_block, NE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, e.data_ptr<float>(), h.data_ptr<float>());
k_6_fop_1<<<num_ne_block, NE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, h.data_ptr<float>(), lr.data_ptr<float>(), em.data_ptr<float>());
k_6_fop_2<<<num_ne_block, NE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, h.data_ptr<float>(), em.data_ptr<float>(), ho.data_ptr<float>());
k_6_fop_3<<<num_ne_block, NE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, ho.data_ptr<float>(), es.data_ptr<float>());
k_6_fop_4<<<num_ngd_block, NGD_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, ho.data_ptr<float>(), f.data_ptr<float>(), es.data_ptr<float>(), fo.data_ptr<float>());
}
__global__ void k_7_fop_0(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* e, float* h, float* lr, float* em, float* ho, float* es) {
int lane_id = threadIdx.x % WARP_SIZE;
int row_tmp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
if (row_tmp >= numnodes) return;
{ for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];
}
}
{ float res_tmp = -99999;
for (int nnz_tmp = RowPtr[row_tmp]+lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) em[row_tmp * 1 + 0] = res_tmp;
}
{ for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);
}
}
{ float res_tmp = 0;
for (int nnz_tmp = RowPtr[row_tmp]+lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) es[row_tmp * 1 + 0] = res_tmp;
}
}
__global__ void k_7_fop_1(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* ho, float* f, float* es, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (global_wid >= *ng_n) return;
neighbor_group ng_tmp = ng_list[global_wid];
int row_tmp = ng_tmp.row_st;
int col_st = ng_tmp.col_st;
int col_ed = ng_tmp.col_st + NG_SIZE;
if (col_ed >= RowPtr[row_tmp + 1]) col_ed = RowPtr[row_tmp + 1];
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = col_st; nnz_tmp < col_ed; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
atomicAdd(&fo[row_tmp * featlen + k_tmp], res_tmp);
}
}
void gat_kernel_7(int64_t info_addr, torch::Tensor RowPtr, torch::Tensor ColIdx, int64_t featlen, torch::Tensor f, torch::Tensor fo, torch::Tensor we, torch::Tensor lr, torch::Tensor e, torch::Tensor em, torch::Tensor es, torch::Tensor h, torch::Tensor ho) {
kg_info* info_ = (kg_info*)info_addr;
if (info_ == nullptr) {
  throw std::runtime_error("info_ is null in gat_kernel_7");
}
if (info_->rp_info == nullptr || info_->rp_n == nullptr || info_->ep_info == nullptr || info_->ep_n == nullptr || info_->ng_info == nullptr || info_->ng_n == nullptr) {
  throw std::runtime_error("Some preprocessing pointers are null in gat_kernel_7");
}
int numnodes = RowPtr.size(0) - 1;
int numedges = ColIdx.size(0);
int num_n_block = (numnodes + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
int num_ne_block = (numnodes + NE_BLOCK_WARP_SIZE - 1) / NE_BLOCK_WARP_SIZE;
int num_e_block = info_->ep_n_host;
int num_he_block = info_->rp_n_host;
int num_ngd_block = (info_->ng_n_host + NGD_BLOCK_WARP_SIZE - 1) / NGD_BLOCK_WARP_SIZE;
k_7_fop_0<<<num_ne_block, NE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, e.data_ptr<float>(), h.data_ptr<float>(), lr.data_ptr<float>(), em.data_ptr<float>(), ho.data_ptr<float>(), es.data_ptr<float>());
k_7_fop_1<<<num_ngd_block, NGD_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, ho.data_ptr<float>(), f.data_ptr<float>(), es.data_ptr<float>(), fo.data_ptr<float>());
}
__global__ void k_8_fop_0(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* e, float* h) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *ep_n) return;
row_panel info_tmp = ep_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (info_tmp.col_st == -1){ 
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
int row_tmp = info_tmp.row_st;
for (int nnz_tmp = col_st + threadIdx.x; nnz_tmp < col_ed; nnz_tmp += blockDim.x) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];
}
}
else {
int row_tmp = info_tmp.row_st;
for (int nnz_tmp = info_tmp.col_st + threadIdx.x; nnz_tmp < info_tmp.col_ed; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];}
}
}
__global__ void k_8_fop_1(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* h, float* lr, float* em) {
int lane_id = threadIdx.x % WARP_SIZE;
int row_tmp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
if (row_tmp >= numnodes) return;
float res_tmp = -99999;
for (int nnz_tmp = RowPtr[row_tmp]+lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) em[row_tmp * 1 + 0] = res_tmp;
}
__global__ void k_8_fop_2(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* h, float* em, float* ho) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *ep_n) return;
row_panel info_tmp = ep_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (info_tmp.col_st == -1){ 
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
int row_tmp = info_tmp.row_st;
for (int nnz_tmp = col_st + threadIdx.x; nnz_tmp < col_ed; nnz_tmp += blockDim.x) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);
}
}
else {
int row_tmp = info_tmp.row_st;
for (int nnz_tmp = info_tmp.col_st + threadIdx.x; nnz_tmp < info_tmp.col_ed; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);}
}
}
__global__ void k_8_fop_3(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* ho, float* es) {
int lane_id = threadIdx.x % WARP_SIZE;
int row_tmp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
if (row_tmp >= numnodes) return;
float res_tmp = 0;
for (int nnz_tmp = RowPtr[row_tmp]+lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) es[row_tmp * 1 + 0] = res_tmp;
}
__global__ void k_8_fop_4(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* ho, float* f, float* es, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
int row_tmp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
if (row_tmp >= numnodes) return;
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = RowPtr[row_tmp]; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
fo[row_tmp * featlen + k_tmp] = res_tmp;
}
}
void gat_kernel_8(int64_t info_addr, torch::Tensor RowPtr, torch::Tensor ColIdx, int64_t featlen, torch::Tensor f, torch::Tensor fo, torch::Tensor we, torch::Tensor lr, torch::Tensor e, torch::Tensor em, torch::Tensor es, torch::Tensor h, torch::Tensor ho) {
kg_info* info_ = (kg_info*)info_addr;
if (info_ == nullptr) {
  throw std::runtime_error("info_ is null in gat_kernel_8");
}
if (info_->rp_info == nullptr || info_->rp_n == nullptr || info_->ep_info == nullptr || info_->ep_n == nullptr || info_->ng_info == nullptr || info_->ng_n == nullptr) {
  throw std::runtime_error("Some preprocessing pointers are null in gat_kernel_8");
}
int numnodes = RowPtr.size(0) - 1;
int numedges = ColIdx.size(0);
int num_n_block = (numnodes + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
int num_ne_block = (numnodes + NE_BLOCK_WARP_SIZE - 1) / NE_BLOCK_WARP_SIZE;
int num_e_block = info_->ep_n_host;
int num_he_block = info_->rp_n_host;
int num_ngd_block = (info_->ng_n_host + NGD_BLOCK_WARP_SIZE - 1) / NGD_BLOCK_WARP_SIZE;
k_8_fop_0<<<num_e_block, E_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, e.data_ptr<float>(), h.data_ptr<float>());
k_8_fop_1<<<num_ne_block, NE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, h.data_ptr<float>(), lr.data_ptr<float>(), em.data_ptr<float>());
k_8_fop_2<<<num_e_block, E_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, h.data_ptr<float>(), em.data_ptr<float>(), ho.data_ptr<float>());
k_8_fop_3<<<num_ne_block, NE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, ho.data_ptr<float>(), es.data_ptr<float>());
k_8_fop_4<<<num_ne_block, NE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, ho.data_ptr<float>(), f.data_ptr<float>(), es.data_ptr<float>(), fo.data_ptr<float>());
}
__global__ void k9_fop0_dp0(int *RowPtr, int *ColIdx, int row_tmp, int featlen, float* e, float* h) {
int lane_id = threadIdx.x % WARP_SIZE;
int idx_st = RowPtr[row_tmp] + blockIdx.x * DP_NNZ_PER_BLOCK + threadIdx.x / WARP_SIZE * DP_NNZ_PER_WARP;
int idx_ed = idx_st + DP_NNZ_PER_WARP;
if (idx_ed > RowPtr[row_tmp + 1]) idx_ed = RowPtr[row_tmp + 1];
for (int nnz_tmp = idx_st + lane_id; nnz_tmp < idx_ed; nnz_tmp += WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];}
}
__global__ void k9_fop0_dp2(int *RowPtr, int *ColIdx, int row_tmp, int featlen, float* h, float* em, float* ho, float* es) {
int lane_id = threadIdx.x % WARP_SIZE;
int idx_st = RowPtr[row_tmp] + blockIdx.x * DP_NNZ_PER_BLOCK + threadIdx.x / WARP_SIZE * DP_NNZ_PER_WARP;
int idx_ed = idx_st + DP_NNZ_PER_WARP;
if (idx_ed > RowPtr[row_tmp + 1]) idx_ed = RowPtr[row_tmp + 1];
for (int nnz_tmp = idx_st + lane_id; nnz_tmp < idx_ed; nnz_tmp += WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);}
float res_tmp = 0;
for (int nnz_tmp = idx_st + lane_id; nnz_tmp < idx_ed; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) {
atomicAdd(&es[row_tmp * 1 + 0], res_tmp);
}
}
__global__ void k_9_fop_0(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* e, float* h, float* lr, float* em, float* ho, float* es) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *info_n) return;
row_panel info_tmp = info_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
int row_tmp_ne_st = info_tmp.row_st + local_wid;
int row_tmp_ne_ed = info_tmp.row_ed;
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
if (info_tmp.col_st == -1) {
{ { int row_tmp = info_tmp.row_st;
for (int nnz_tmp = col_st + threadIdx.x; nnz_tmp < col_ed; nnz_tmp += HE_BLOCK_SIZE) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];
}
}
}
__syncthreads();
{ for (int row_tmp = row_tmp_ne_st; row_tmp < row_tmp_ne_ed; row_tmp += HE_BLOCK_WARP_SIZE) {
float res_tmp = -99999;
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;
}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) em[row_tmp * 1 + 0] = res_tmp;
}
}
__syncthreads();
{ { int row_tmp = info_tmp.row_st;
for (int nnz_tmp = col_st + threadIdx.x; nnz_tmp < col_ed; nnz_tmp += HE_BLOCK_SIZE) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);
}
}
}
__syncthreads();
{ for (int row_tmp = row_tmp_ne_st; row_tmp < row_tmp_ne_ed; row_tmp += HE_BLOCK_WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];
}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) es[row_tmp * 1 + 0] = res_tmp;
}
}
__syncthreads();
}
else {
int row_tmp = info_tmp.row_st;
int nnzs = RowPtr[row_tmp + 1] - RowPtr[row_tmp];
int dynamic_blocks = (nnzs + DP_NNZ_PER_BLOCK - 1) / DP_NNZ_PER_BLOCK;
if (threadIdx.x == 0) k9_fop0_dp0<<<dynamic_blocks, DP_BLOCK_SIZE>>>(RowPtr, ColIdx, row_tmp, featlen, e, h);
{
float res_tmp = -99999;
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) {
atomicMax_float(&em[row_tmp * 1 + 0], res_tmp);
}
}
if (threadIdx.x == 0) k9_fop0_dp2<<<dynamic_blocks, DP_BLOCK_SIZE>>>(RowPtr, ColIdx, row_tmp, featlen, h, em, ho, es);
}
}
__global__ void k_9_fop_1(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* ho, float* f, float* es, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
int row_tmp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
if (row_tmp >= numnodes) return;
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = RowPtr[row_tmp]; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
fo[row_tmp * featlen + k_tmp] = res_tmp;
}
}
void gat_kernel_9(int64_t info_addr, torch::Tensor RowPtr, torch::Tensor ColIdx, int64_t featlen, torch::Tensor f, torch::Tensor fo, torch::Tensor we, torch::Tensor lr, torch::Tensor e, torch::Tensor em, torch::Tensor es, torch::Tensor h, torch::Tensor ho) {
kg_info* info_ = (kg_info*)info_addr;
if (info_ == nullptr) {
  throw std::runtime_error("info_ is null in gat_kernel_9");
}
if (info_->rp_info == nullptr || info_->rp_n == nullptr || info_->ep_info == nullptr || info_->ep_n == nullptr || info_->ng_info == nullptr || info_->ng_n == nullptr) {
  throw std::runtime_error("Some preprocessing pointers are null in gat_kernel_9");
}
int numnodes = RowPtr.size(0) - 1;
int numedges = ColIdx.size(0);
int num_n_block = (numnodes + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
int num_ne_block = (numnodes + NE_BLOCK_WARP_SIZE - 1) / NE_BLOCK_WARP_SIZE;
int num_e_block = info_->ep_n_host;
int num_he_block = info_->rp_n_host;
int num_ngd_block = (info_->ng_n_host + NGD_BLOCK_WARP_SIZE - 1) / NGD_BLOCK_WARP_SIZE;
k_9_fop_0<<<num_he_block, HE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, e.data_ptr<float>(), h.data_ptr<float>(), lr.data_ptr<float>(), em.data_ptr<float>(), ho.data_ptr<float>(), es.data_ptr<float>());
k_9_fop_1<<<num_ne_block, NE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, ho.data_ptr<float>(), f.data_ptr<float>(), es.data_ptr<float>(), fo.data_ptr<float>());
}
__global__ void k10_fop0_dp4(int *RowPtr, int *ColIdx, int row_tmp, int featlen, float* ho, float* f, float* es, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
int idx_st = RowPtr[row_tmp] + blockIdx.x * DP_NNZ_PER_BLOCK + threadIdx.x / WARP_SIZE * DP_NNZ_PER_WARP;
int idx_ed = idx_st + DP_NNZ_PER_WARP;
if (idx_ed > RowPtr[row_tmp + 1]) idx_ed = RowPtr[row_tmp + 1];
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = idx_st; nnz_tmp < idx_ed; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];
}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
atomicAdd(&fo[row_tmp * featlen + k_tmp], res_tmp);
}
}
__global__ void k_10_fop_0(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* e, float* h, float* lr, float* em, float* ho, float* es, float* f, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *info_n) return;
row_panel info_tmp = info_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
int row_tmp_ne_st = info_tmp.row_st + local_wid;
int row_tmp_ne_ed = info_tmp.row_ed;
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
if (info_tmp.col_st == -1) {
{ { int row_tmp = info_tmp.row_st;
for (int nnz_tmp = col_st + threadIdx.x; nnz_tmp < col_ed; nnz_tmp += HE_BLOCK_SIZE) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];
}
}
}
__syncthreads();
{ for (int row_tmp = row_tmp_ne_st; row_tmp < row_tmp_ne_ed; row_tmp += HE_BLOCK_WARP_SIZE) {
float res_tmp = -99999;
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;
}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) em[row_tmp * 1 + 0] = res_tmp;
}
}
__syncthreads();
{ { int row_tmp = info_tmp.row_st;
for (int nnz_tmp = col_st + threadIdx.x; nnz_tmp < col_ed; nnz_tmp += HE_BLOCK_SIZE) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);
}
}
}
__syncthreads();
{ for (int row_tmp = row_tmp_ne_st; row_tmp < row_tmp_ne_ed; row_tmp += HE_BLOCK_WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];
}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) es[row_tmp * 1 + 0] = res_tmp;
}
}
__syncthreads();
{ for (int row_tmp = row_tmp_ne_st; row_tmp < row_tmp_ne_ed; row_tmp += HE_BLOCK_WARP_SIZE) {
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = RowPtr[row_tmp]; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];
}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
fo[row_tmp * featlen + k_tmp] = res_tmp;
}
}
}
__syncthreads();
}
else {
int row_tmp = info_tmp.row_st;
int nnzs = RowPtr[row_tmp + 1] - RowPtr[row_tmp];
int dynamic_blocks = (nnzs + DP_NNZ_PER_BLOCK - 1) / DP_NNZ_PER_BLOCK;
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];}__syncthreads();
{
float res_tmp = -99999;
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) {
atomicMax_float(&em[row_tmp * 1 + 0], res_tmp);
}
}__syncthreads();
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);}__syncthreads();
{
float res_tmp = 0;
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) {
atomicAdd(&es[row_tmp * 1 + 0], res_tmp);
}
}__syncthreads();
if (threadIdx.x == 0) k10_fop0_dp4<<<dynamic_blocks, DP_BLOCK_SIZE>>>(RowPtr, ColIdx, row_tmp, featlen, ho, f, es, fo);
}
}
void gat_kernel_10(int64_t info_addr, torch::Tensor RowPtr, torch::Tensor ColIdx, int64_t featlen, torch::Tensor f, torch::Tensor fo, torch::Tensor we, torch::Tensor lr, torch::Tensor e, torch::Tensor em, torch::Tensor es, torch::Tensor h, torch::Tensor ho) {
kg_info* info_ = (kg_info*)info_addr;
if (info_ == nullptr) {
  throw std::runtime_error("info_ is null in gat_kernel_10");
}
if (info_->rp_info == nullptr || info_->rp_n == nullptr || info_->ep_info == nullptr || info_->ep_n == nullptr || info_->ng_info == nullptr || info_->ng_n == nullptr) {
  throw std::runtime_error("Some preprocessing pointers are null in gat_kernel_10");
}
int numnodes = RowPtr.size(0) - 1;
int numedges = ColIdx.size(0);
int num_n_block = (numnodes + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
int num_ne_block = (numnodes + NE_BLOCK_WARP_SIZE - 1) / NE_BLOCK_WARP_SIZE;
int num_e_block = info_->ep_n_host;
int num_he_block = info_->rp_n_host;
int num_ngd_block = (info_->ng_n_host + NGD_BLOCK_WARP_SIZE - 1) / NGD_BLOCK_WARP_SIZE;
k_10_fop_0<<<num_he_block, HE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, e.data_ptr<float>(), h.data_ptr<float>(), lr.data_ptr<float>(), em.data_ptr<float>(), ho.data_ptr<float>(), es.data_ptr<float>(), f.data_ptr<float>(), fo.data_ptr<float>());
}
__global__ void k_11_fop_0(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* e, float* h) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *ep_n) return;
row_panel info_tmp = ep_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (info_tmp.col_st == -1){ 
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
int row_tmp = info_tmp.row_st;
for (int nnz_tmp = col_st + threadIdx.x; nnz_tmp < col_ed; nnz_tmp += blockDim.x) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];
}
}
else {
int row_tmp = info_tmp.row_st;
for (int nnz_tmp = info_tmp.col_st + threadIdx.x; nnz_tmp < info_tmp.col_ed; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];}
}
}
__global__ void k_11_fop_1(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* h, float* lr, float* em) {
int lane_id = threadIdx.x % WARP_SIZE;
int row_tmp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
if (row_tmp >= numnodes) return;
float res_tmp = -99999;
for (int nnz_tmp = RowPtr[row_tmp]+lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) em[row_tmp * 1 + 0] = res_tmp;
}
__global__ void k_11_fop_2(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* h, float* em, float* ho) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *ep_n) return;
row_panel info_tmp = ep_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (info_tmp.col_st == -1){ 
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
int row_tmp = info_tmp.row_st;
for (int nnz_tmp = col_st + threadIdx.x; nnz_tmp < col_ed; nnz_tmp += blockDim.x) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);
}
}
else {
int row_tmp = info_tmp.row_st;
for (int nnz_tmp = info_tmp.col_st + threadIdx.x; nnz_tmp < info_tmp.col_ed; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);}
}
}
__global__ void k_11_fop_3(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* ho, float* es) {
int lane_id = threadIdx.x % WARP_SIZE;
int row_tmp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
if (row_tmp >= numnodes) return;
float res_tmp = 0;
for (int nnz_tmp = RowPtr[row_tmp]+lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) es[row_tmp * 1 + 0] = res_tmp;
}
__global__ void k_11_fop_4(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* ho, float* f, float* es, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *ep_n) return;
row_panel info_tmp = ep_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (info_tmp.col_st == -1){ 
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
int nnz_per_warp = (nnzs + E_BLOCK_WARP_SIZE - 1) / E_BLOCK_WARP_SIZE;
int warp_col_st = col_st + nnz_per_warp * local_wid;
int warp_col_ed = (warp_col_st + nnz_per_warp < col_ed)? warp_col_st + nnz_per_warp: col_ed;
int warp_row_st = info_tmp.row_st, warp_row_ed;
while (warp_row_st < info_tmp.row_ed - 1 && warp_col_st > RowPtr[warp_row_st + 1]) {
warp_row_st++;
}
warp_row_ed = warp_row_st;
while (warp_row_ed < info_tmp.row_ed && warp_col_ed > RowPtr[warp_row_ed]) {
warp_row_ed++;
}
if (warp_col_ed <= col_ed) {
for (int row_tmp = warp_row_st; row_tmp < warp_row_ed; row_tmp++) {
int current_col_st = RowPtr[row_tmp];
int current_col_ed = RowPtr[row_tmp + 1];
if (row_tmp == warp_row_st) current_col_st = warp_col_st;
if (row_tmp == warp_row_ed - 1) current_col_ed = warp_col_ed;
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = current_col_st; nnz_tmp < current_col_ed; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
atomicAdd(&fo[row_tmp * featlen + k_tmp], res_tmp);
}}}
}
else {
int row_tmp = info_tmp.row_st;
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = info_tmp.col_st + local_wid; nnz_tmp < info_tmp.col_ed; nnz_tmp+=HE_BLOCK_WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];
}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
atomicAdd(&fo[row_tmp * featlen + k_tmp], res_tmp);
}
}
}
void gat_kernel_11(int64_t info_addr, torch::Tensor RowPtr, torch::Tensor ColIdx, int64_t featlen, torch::Tensor f, torch::Tensor fo, torch::Tensor we, torch::Tensor lr, torch::Tensor e, torch::Tensor em, torch::Tensor es, torch::Tensor h, torch::Tensor ho) {
kg_info* info_ = (kg_info*)info_addr;
if (info_ == nullptr) {
  throw std::runtime_error("info_ is null in gat_kernel_11");
}
if (info_->rp_info == nullptr || info_->rp_n == nullptr || info_->ep_info == nullptr || info_->ep_n == nullptr || info_->ng_info == nullptr || info_->ng_n == nullptr) {
  throw std::runtime_error("Some preprocessing pointers are null in gat_kernel_11");
}
int numnodes = RowPtr.size(0) - 1;
int numedges = ColIdx.size(0);
int num_n_block = (numnodes + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
int num_ne_block = (numnodes + NE_BLOCK_WARP_SIZE - 1) / NE_BLOCK_WARP_SIZE;
int num_e_block = info_->ep_n_host;
int num_he_block = info_->rp_n_host;
int num_ngd_block = (info_->ng_n_host + NGD_BLOCK_WARP_SIZE - 1) / NGD_BLOCK_WARP_SIZE;
k_11_fop_0<<<num_e_block, E_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, e.data_ptr<float>(), h.data_ptr<float>());
k_11_fop_1<<<num_ne_block, NE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, h.data_ptr<float>(), lr.data_ptr<float>(), em.data_ptr<float>());
k_11_fop_2<<<num_e_block, E_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, h.data_ptr<float>(), em.data_ptr<float>(), ho.data_ptr<float>());
k_11_fop_3<<<num_ne_block, NE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, ho.data_ptr<float>(), es.data_ptr<float>());
k_11_fop_4<<<num_e_block, E_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, ho.data_ptr<float>(), f.data_ptr<float>(), es.data_ptr<float>(), fo.data_ptr<float>());
}
__global__ void k12_fop0_dp0(int *RowPtr, int *ColIdx, int row_tmp, int featlen, float* e, float* h) {
int lane_id = threadIdx.x % WARP_SIZE;
int idx_st = RowPtr[row_tmp] + blockIdx.x * DP_NNZ_PER_BLOCK + threadIdx.x / WARP_SIZE * DP_NNZ_PER_WARP;
int idx_ed = idx_st + DP_NNZ_PER_WARP;
if (idx_ed > RowPtr[row_tmp + 1]) idx_ed = RowPtr[row_tmp + 1];
for (int nnz_tmp = idx_st + lane_id; nnz_tmp < idx_ed; nnz_tmp += WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];}
}
__global__ void k12_fop0_dp2(int *RowPtr, int *ColIdx, int row_tmp, int featlen, float* h, float* em, float* ho, float* es) {
int lane_id = threadIdx.x % WARP_SIZE;
int idx_st = RowPtr[row_tmp] + blockIdx.x * DP_NNZ_PER_BLOCK + threadIdx.x / WARP_SIZE * DP_NNZ_PER_WARP;
int idx_ed = idx_st + DP_NNZ_PER_WARP;
if (idx_ed > RowPtr[row_tmp + 1]) idx_ed = RowPtr[row_tmp + 1];
for (int nnz_tmp = idx_st + lane_id; nnz_tmp < idx_ed; nnz_tmp += WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);}
float res_tmp = 0;
for (int nnz_tmp = idx_st + lane_id; nnz_tmp < idx_ed; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) {
atomicAdd(&es[row_tmp * 1 + 0], res_tmp);
}
}
__global__ void k_12_fop_0(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* e, float* h, float* lr, float* em, float* ho, float* es) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *info_n) return;
row_panel info_tmp = info_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
int row_tmp_ne_st = info_tmp.row_st + local_wid;
int row_tmp_ne_ed = info_tmp.row_ed;
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
if (info_tmp.col_st == -1) {
{ { int row_tmp = info_tmp.row_st;
for (int nnz_tmp = col_st + threadIdx.x; nnz_tmp < col_ed; nnz_tmp += HE_BLOCK_SIZE) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];
}
}
}
__syncthreads();
{ for (int row_tmp = row_tmp_ne_st; row_tmp < row_tmp_ne_ed; row_tmp += HE_BLOCK_WARP_SIZE) {
float res_tmp = -99999;
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;
}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) em[row_tmp * 1 + 0] = res_tmp;
}
}
__syncthreads();
{ { int row_tmp = info_tmp.row_st;
for (int nnz_tmp = col_st + threadIdx.x; nnz_tmp < col_ed; nnz_tmp += HE_BLOCK_SIZE) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);
}
}
}
__syncthreads();
{ for (int row_tmp = row_tmp_ne_st; row_tmp < row_tmp_ne_ed; row_tmp += HE_BLOCK_WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];
}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) es[row_tmp * 1 + 0] = res_tmp;
}
}
__syncthreads();
}
else {
int row_tmp = info_tmp.row_st;
int nnzs = RowPtr[row_tmp + 1] - RowPtr[row_tmp];
int dynamic_blocks = (nnzs + DP_NNZ_PER_BLOCK - 1) / DP_NNZ_PER_BLOCK;
if (threadIdx.x == 0) k12_fop0_dp0<<<dynamic_blocks, DP_BLOCK_SIZE>>>(RowPtr, ColIdx, row_tmp, featlen, e, h);
{
float res_tmp = -99999;
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) {
atomicMax_float(&em[row_tmp * 1 + 0], res_tmp);
}
}
if (threadIdx.x == 0) k12_fop0_dp2<<<dynamic_blocks, DP_BLOCK_SIZE>>>(RowPtr, ColIdx, row_tmp, featlen, h, em, ho, es);
}
}
__global__ void k_12_fop_1(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* ho, float* f, float* es, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *ep_n) return;
row_panel info_tmp = ep_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (info_tmp.col_st == -1){ 
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
int nnz_per_warp = (nnzs + E_BLOCK_WARP_SIZE - 1) / E_BLOCK_WARP_SIZE;
int warp_col_st = col_st + nnz_per_warp * local_wid;
int warp_col_ed = (warp_col_st + nnz_per_warp < col_ed)? warp_col_st + nnz_per_warp: col_ed;
int warp_row_st = info_tmp.row_st, warp_row_ed;
while (warp_row_st < info_tmp.row_ed - 1 && warp_col_st > RowPtr[warp_row_st + 1]) {
warp_row_st++;
}
warp_row_ed = warp_row_st;
while (warp_row_ed < info_tmp.row_ed && warp_col_ed > RowPtr[warp_row_ed]) {
warp_row_ed++;
}
if (warp_col_ed <= col_ed) {
for (int row_tmp = warp_row_st; row_tmp < warp_row_ed; row_tmp++) {
int current_col_st = RowPtr[row_tmp];
int current_col_ed = RowPtr[row_tmp + 1];
if (row_tmp == warp_row_st) current_col_st = warp_col_st;
if (row_tmp == warp_row_ed - 1) current_col_ed = warp_col_ed;
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = current_col_st; nnz_tmp < current_col_ed; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
atomicAdd(&fo[row_tmp * featlen + k_tmp], res_tmp);
}}}
}
else {
int row_tmp = info_tmp.row_st;
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = info_tmp.col_st + local_wid; nnz_tmp < info_tmp.col_ed; nnz_tmp+=HE_BLOCK_WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];
}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
atomicAdd(&fo[row_tmp * featlen + k_tmp], res_tmp);
}
}
}
void gat_kernel_12(int64_t info_addr, torch::Tensor RowPtr, torch::Tensor ColIdx, int64_t featlen, torch::Tensor f, torch::Tensor fo, torch::Tensor we, torch::Tensor lr, torch::Tensor e, torch::Tensor em, torch::Tensor es, torch::Tensor h, torch::Tensor ho) {
kg_info* info_ = (kg_info*)info_addr;
if (info_ == nullptr) {
  throw std::runtime_error("info_ is null in gat_kernel_12");
}
if (info_->rp_info == nullptr || info_->rp_n == nullptr || info_->ep_info == nullptr || info_->ep_n == nullptr || info_->ng_info == nullptr || info_->ng_n == nullptr) {
  throw std::runtime_error("Some preprocessing pointers are null in gat_kernel_12");
}
int numnodes = RowPtr.size(0) - 1;
int numedges = ColIdx.size(0);
int num_n_block = (numnodes + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
int num_ne_block = (numnodes + NE_BLOCK_WARP_SIZE - 1) / NE_BLOCK_WARP_SIZE;
int num_e_block = info_->ep_n_host;
int num_he_block = info_->rp_n_host;
int num_ngd_block = (info_->ng_n_host + NGD_BLOCK_WARP_SIZE - 1) / NGD_BLOCK_WARP_SIZE;
k_12_fop_0<<<num_he_block, HE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, e.data_ptr<float>(), h.data_ptr<float>(), lr.data_ptr<float>(), em.data_ptr<float>(), ho.data_ptr<float>(), es.data_ptr<float>());
k_12_fop_1<<<num_e_block, E_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, ho.data_ptr<float>(), f.data_ptr<float>(), es.data_ptr<float>(), fo.data_ptr<float>());
}
__global__ void k13_fop0_dp4(int *RowPtr, int *ColIdx, int row_tmp, int featlen, float* ho, float* f, float* es, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
int idx_st = RowPtr[row_tmp] + blockIdx.x * DP_NNZ_PER_BLOCK + threadIdx.x / WARP_SIZE * DP_NNZ_PER_WARP;
int idx_ed = idx_st + DP_NNZ_PER_WARP;
if (idx_ed > RowPtr[row_tmp + 1]) idx_ed = RowPtr[row_tmp + 1];
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = idx_st; nnz_tmp < idx_ed; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];
}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
atomicAdd(&fo[row_tmp * featlen + k_tmp], res_tmp);
}
}
__global__ void k_13_fop_0(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* e, float* h, float* lr, float* em, float* ho, float* es, float* f, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *info_n) return;
row_panel info_tmp = info_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
int row_tmp_ne_st = info_tmp.row_st + local_wid;
int row_tmp_ne_ed = info_tmp.row_ed;
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
int nnz_per_warp = (nnzs + HE_BLOCK_WARP_SIZE - 1) / HE_BLOCK_WARP_SIZE;
int warp_col_st = col_st + nnz_per_warp * local_wid;
int warp_col_ed = (warp_col_st + nnz_per_warp < col_ed)? warp_col_st + nnz_per_warp: col_ed;
int warp_row_st = info_tmp.row_st, warp_row_ed;
while (warp_row_st < info_tmp.row_ed - 1 && warp_col_st > RowPtr[warp_row_st + 1]) {
warp_row_st++;
}
warp_row_ed = warp_row_st;
while (warp_row_ed < info_tmp.row_ed && warp_col_ed > RowPtr[warp_row_ed]) {
warp_row_ed++;
}
if (info_tmp.col_st == -1) {
{ { int row_tmp = info_tmp.row_st;
for (int nnz_tmp = col_st + threadIdx.x; nnz_tmp < col_ed; nnz_tmp += HE_BLOCK_SIZE) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];
}
}
}
__syncthreads();
{ for (int row_tmp = row_tmp_ne_st; row_tmp < row_tmp_ne_ed; row_tmp += HE_BLOCK_WARP_SIZE) {
float res_tmp = -99999;
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;
}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) em[row_tmp * 1 + 0] = res_tmp;
}
}
__syncthreads();
{ { int row_tmp = info_tmp.row_st;
for (int nnz_tmp = col_st + threadIdx.x; nnz_tmp < col_ed; nnz_tmp += HE_BLOCK_SIZE) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);
}
}
}
__syncthreads();
{ for (int row_tmp = row_tmp_ne_st; row_tmp < row_tmp_ne_ed; row_tmp += HE_BLOCK_WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];
}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) es[row_tmp * 1 + 0] = res_tmp;
}
}
__syncthreads();
{ if (warp_col_ed <= col_ed) {
for (int row_tmp = warp_row_st; row_tmp < warp_row_ed; row_tmp++) {
int current_col_st = RowPtr[row_tmp];
int current_col_ed = RowPtr[row_tmp + 1];
if (row_tmp == warp_row_st) current_col_st = warp_col_st;
if (row_tmp == warp_row_ed - 1) current_col_ed = warp_col_ed;
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = current_col_st; nnz_tmp < current_col_ed; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];
}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
atomicAdd(&fo[row_tmp * featlen + k_tmp], res_tmp);
}}}
}
__syncthreads();
}
else {
int row_tmp = info_tmp.row_st;
int nnzs = RowPtr[row_tmp + 1] - RowPtr[row_tmp];
int dynamic_blocks = (nnzs + DP_NNZ_PER_BLOCK - 1) / DP_NNZ_PER_BLOCK;
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];}__syncthreads();
{
float res_tmp = -99999;
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) {
atomicMax_float(&em[row_tmp * 1 + 0], res_tmp);
}
}__syncthreads();
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);}__syncthreads();
{
float res_tmp = 0;
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) {
atomicAdd(&es[row_tmp * 1 + 0], res_tmp);
}
}__syncthreads();
if (threadIdx.x == 0) k13_fop0_dp4<<<dynamic_blocks, DP_BLOCK_SIZE>>>(RowPtr, ColIdx, row_tmp, featlen, ho, f, es, fo);
}
}
void gat_kernel_13(int64_t info_addr, torch::Tensor RowPtr, torch::Tensor ColIdx, int64_t featlen, torch::Tensor f, torch::Tensor fo, torch::Tensor we, torch::Tensor lr, torch::Tensor e, torch::Tensor em, torch::Tensor es, torch::Tensor h, torch::Tensor ho) {
kg_info* info_ = (kg_info*)info_addr;
if (info_ == nullptr) {
  throw std::runtime_error("info_ is null in gat_kernel_13");
}
if (info_->rp_info == nullptr || info_->rp_n == nullptr || info_->ep_info == nullptr || info_->ep_n == nullptr || info_->ng_info == nullptr || info_->ng_n == nullptr) {
  throw std::runtime_error("Some preprocessing pointers are null in gat_kernel_13");
}
int numnodes = RowPtr.size(0) - 1;
int numedges = ColIdx.size(0);
int num_n_block = (numnodes + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
int num_ne_block = (numnodes + NE_BLOCK_WARP_SIZE - 1) / NE_BLOCK_WARP_SIZE;
int num_e_block = info_->ep_n_host;
int num_he_block = info_->rp_n_host;
int num_ngd_block = (info_->ng_n_host + NGD_BLOCK_WARP_SIZE - 1) / NGD_BLOCK_WARP_SIZE;
k_13_fop_0<<<num_he_block, HE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, e.data_ptr<float>(), h.data_ptr<float>(), lr.data_ptr<float>(), em.data_ptr<float>(), ho.data_ptr<float>(), es.data_ptr<float>(), f.data_ptr<float>(), fo.data_ptr<float>());
}
__global__ void k_14_fop_0(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* e, float* h) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *ep_n) return;
row_panel info_tmp = ep_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (info_tmp.col_st == -1){ 
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
int row_tmp = info_tmp.row_st;
for (int nnz_tmp = col_st + threadIdx.x; nnz_tmp < col_ed; nnz_tmp += blockDim.x) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];
}
}
else {
int row_tmp = info_tmp.row_st;
for (int nnz_tmp = info_tmp.col_st + threadIdx.x; nnz_tmp < info_tmp.col_ed; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];}
}
}
__global__ void k_14_fop_1(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* h, float* lr, float* em) {
int lane_id = threadIdx.x % WARP_SIZE;
int row_tmp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
if (row_tmp >= numnodes) return;
float res_tmp = -99999;
for (int nnz_tmp = RowPtr[row_tmp]+lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) em[row_tmp * 1 + 0] = res_tmp;
}
__global__ void k_14_fop_2(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* h, float* em, float* ho) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *ep_n) return;
row_panel info_tmp = ep_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (info_tmp.col_st == -1){ 
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
int row_tmp = info_tmp.row_st;
for (int nnz_tmp = col_st + threadIdx.x; nnz_tmp < col_ed; nnz_tmp += blockDim.x) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);
}
}
else {
int row_tmp = info_tmp.row_st;
for (int nnz_tmp = info_tmp.col_st + threadIdx.x; nnz_tmp < info_tmp.col_ed; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);}
}
}
__global__ void k_14_fop_3(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* ho, float* es) {
int lane_id = threadIdx.x % WARP_SIZE;
int row_tmp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
if (row_tmp >= numnodes) return;
float res_tmp = 0;
for (int nnz_tmp = RowPtr[row_tmp]+lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) es[row_tmp * 1 + 0] = res_tmp;
}
__global__ void k_14_fop_4(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* ho, float* f, float* es, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (global_wid >= *ng_n) return;
neighbor_group ng_tmp = ng_list[global_wid];
int row_tmp = ng_tmp.row_st;
int col_st = ng_tmp.col_st;
int col_ed = ng_tmp.col_st + NG_SIZE;
if (col_ed >= RowPtr[row_tmp + 1]) col_ed = RowPtr[row_tmp + 1];
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = col_st; nnz_tmp < col_ed; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
atomicAdd(&fo[row_tmp * featlen + k_tmp], res_tmp);
}
}
void gat_kernel_14(int64_t info_addr, torch::Tensor RowPtr, torch::Tensor ColIdx, int64_t featlen, torch::Tensor f, torch::Tensor fo, torch::Tensor we, torch::Tensor lr, torch::Tensor e, torch::Tensor em, torch::Tensor es, torch::Tensor h, torch::Tensor ho) {
kg_info* info_ = (kg_info*)info_addr;
if (info_ == nullptr) {
  throw std::runtime_error("info_ is null in gat_kernel_14");
}
if (info_->rp_info == nullptr || info_->rp_n == nullptr || info_->ep_info == nullptr || info_->ep_n == nullptr || info_->ng_info == nullptr || info_->ng_n == nullptr) {
  throw std::runtime_error("Some preprocessing pointers are null in gat_kernel_14");
}
int numnodes = RowPtr.size(0) - 1;
int numedges = ColIdx.size(0);
int num_n_block = (numnodes + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
int num_ne_block = (numnodes + NE_BLOCK_WARP_SIZE - 1) / NE_BLOCK_WARP_SIZE;
int num_e_block = info_->ep_n_host;
int num_he_block = info_->rp_n_host;
int num_ngd_block = (info_->ng_n_host + NGD_BLOCK_WARP_SIZE - 1) / NGD_BLOCK_WARP_SIZE;
k_14_fop_0<<<num_e_block, E_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, e.data_ptr<float>(), h.data_ptr<float>());
k_14_fop_1<<<num_ne_block, NE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, h.data_ptr<float>(), lr.data_ptr<float>(), em.data_ptr<float>());
k_14_fop_2<<<num_e_block, E_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, h.data_ptr<float>(), em.data_ptr<float>(), ho.data_ptr<float>());
k_14_fop_3<<<num_ne_block, NE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, ho.data_ptr<float>(), es.data_ptr<float>());
k_14_fop_4<<<num_ngd_block, NGD_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, ho.data_ptr<float>(), f.data_ptr<float>(), es.data_ptr<float>(), fo.data_ptr<float>());
}
__global__ void k15_fop0_dp0(int *RowPtr, int *ColIdx, int row_tmp, int featlen, float* e, float* h) {
int lane_id = threadIdx.x % WARP_SIZE;
int idx_st = RowPtr[row_tmp] + blockIdx.x * DP_NNZ_PER_BLOCK + threadIdx.x / WARP_SIZE * DP_NNZ_PER_WARP;
int idx_ed = idx_st + DP_NNZ_PER_WARP;
if (idx_ed > RowPtr[row_tmp + 1]) idx_ed = RowPtr[row_tmp + 1];
for (int nnz_tmp = idx_st + lane_id; nnz_tmp < idx_ed; nnz_tmp += WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];}
}
__global__ void k15_fop0_dp2(int *RowPtr, int *ColIdx, int row_tmp, int featlen, float* h, float* em, float* ho, float* es) {
int lane_id = threadIdx.x % WARP_SIZE;
int idx_st = RowPtr[row_tmp] + blockIdx.x * DP_NNZ_PER_BLOCK + threadIdx.x / WARP_SIZE * DP_NNZ_PER_WARP;
int idx_ed = idx_st + DP_NNZ_PER_WARP;
if (idx_ed > RowPtr[row_tmp + 1]) idx_ed = RowPtr[row_tmp + 1];
for (int nnz_tmp = idx_st + lane_id; nnz_tmp < idx_ed; nnz_tmp += WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);}
float res_tmp = 0;
for (int nnz_tmp = idx_st + lane_id; nnz_tmp < idx_ed; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) {
atomicAdd(&es[row_tmp * 1 + 0], res_tmp);
}
}
__global__ void k_15_fop_0(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* e, float* h, float* lr, float* em, float* ho, float* es) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *info_n) return;
row_panel info_tmp = info_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
int row_tmp_ne_st = info_tmp.row_st + local_wid;
int row_tmp_ne_ed = info_tmp.row_ed;
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
if (info_tmp.col_st == -1) {
{ { int row_tmp = info_tmp.row_st;
for (int nnz_tmp = col_st + threadIdx.x; nnz_tmp < col_ed; nnz_tmp += HE_BLOCK_SIZE) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];
}
}
}
__syncthreads();
{ for (int row_tmp = row_tmp_ne_st; row_tmp < row_tmp_ne_ed; row_tmp += HE_BLOCK_WARP_SIZE) {
float res_tmp = -99999;
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;
}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) em[row_tmp * 1 + 0] = res_tmp;
}
}
__syncthreads();
{ { int row_tmp = info_tmp.row_st;
for (int nnz_tmp = col_st + threadIdx.x; nnz_tmp < col_ed; nnz_tmp += HE_BLOCK_SIZE) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);
}
}
}
__syncthreads();
{ for (int row_tmp = row_tmp_ne_st; row_tmp < row_tmp_ne_ed; row_tmp += HE_BLOCK_WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];
}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) es[row_tmp * 1 + 0] = res_tmp;
}
}
__syncthreads();
}
else {
int row_tmp = info_tmp.row_st;
int nnzs = RowPtr[row_tmp + 1] - RowPtr[row_tmp];
int dynamic_blocks = (nnzs + DP_NNZ_PER_BLOCK - 1) / DP_NNZ_PER_BLOCK;
if (threadIdx.x == 0) k15_fop0_dp0<<<dynamic_blocks, DP_BLOCK_SIZE>>>(RowPtr, ColIdx, row_tmp, featlen, e, h);
{
float res_tmp = -99999;
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) {
atomicMax_float(&em[row_tmp * 1 + 0], res_tmp);
}
}
if (threadIdx.x == 0) k15_fop0_dp2<<<dynamic_blocks, DP_BLOCK_SIZE>>>(RowPtr, ColIdx, row_tmp, featlen, h, em, ho, es);
}
}
__global__ void k_15_fop_1(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* ho, float* f, float* es, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (global_wid >= *ng_n) return;
neighbor_group ng_tmp = ng_list[global_wid];
int row_tmp = ng_tmp.row_st;
int col_st = ng_tmp.col_st;
int col_ed = ng_tmp.col_st + NG_SIZE;
if (col_ed >= RowPtr[row_tmp + 1]) col_ed = RowPtr[row_tmp + 1];
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = col_st; nnz_tmp < col_ed; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
atomicAdd(&fo[row_tmp * featlen + k_tmp], res_tmp);
}
}
void gat_kernel_15(int64_t info_addr, torch::Tensor RowPtr, torch::Tensor ColIdx, int64_t featlen, torch::Tensor f, torch::Tensor fo, torch::Tensor we, torch::Tensor lr, torch::Tensor e, torch::Tensor em, torch::Tensor es, torch::Tensor h, torch::Tensor ho) {
kg_info* info_ = (kg_info*)info_addr;
if (info_ == nullptr) {
  throw std::runtime_error("info_ is null in gat_kernel_15");
}
if (info_->rp_info == nullptr || info_->rp_n == nullptr || info_->ep_info == nullptr || info_->ep_n == nullptr || info_->ng_info == nullptr || info_->ng_n == nullptr) {
  throw std::runtime_error("Some preprocessing pointers are null in gat_kernel_15");
}
int numnodes = RowPtr.size(0) - 1;
int numedges = ColIdx.size(0);
int num_n_block = (numnodes + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
int num_ne_block = (numnodes + NE_BLOCK_WARP_SIZE - 1) / NE_BLOCK_WARP_SIZE;
int num_e_block = info_->ep_n_host;
int num_he_block = info_->rp_n_host;
int num_ngd_block = (info_->ng_n_host + NGD_BLOCK_WARP_SIZE - 1) / NGD_BLOCK_WARP_SIZE;
k_15_fop_0<<<num_he_block, HE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, e.data_ptr<float>(), h.data_ptr<float>(), lr.data_ptr<float>(), em.data_ptr<float>(), ho.data_ptr<float>(), es.data_ptr<float>());
k_15_fop_1<<<num_ngd_block, NGD_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, ho.data_ptr<float>(), f.data_ptr<float>(), es.data_ptr<float>(), fo.data_ptr<float>());
}
__global__ void k_16_fop_0(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* e, float* h) {
int lane_id = threadIdx.x % WARP_SIZE;
int row_tmp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
if (row_tmp >= numnodes) return;
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];
}
}
__global__ void k_16_fop_1(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* h, float* lr, float* em) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *ep_n) return;
row_panel info_tmp = ep_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (info_tmp.col_st == -1){ 
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
using WarpReduce = cub::WarpReduce<float>;
__shared__ typename WarpReduce::TempStorage temp_storage[E_BLOCK_WARP_SIZE];
int row_tmp = info_tmp.row_st;
for (int col_iter = col_st; col_iter < col_ed; col_iter += blockDim.x) {
int nnz_tmp = col_iter + threadIdx.x;
int col_tmp = ColIdx[nnz_tmp];
int new_line_flag = 0; 
float res_tmp = 0;
if (nnz_tmp < col_ed) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
new_line_flag = (RowPtr[row_tmp] == nnz_tmp || lane_id == 0);
res_tmp = kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]);
}
else
res_tmp = -99999;
float reduce_tmp = WarpReduce(temp_storage[local_wid]).HeadSegmentedReduce(res_tmp, new_line_flag, cub::Max());
if (new_line_flag) atomicMax_float(&em[row_tmp * 1 + 0], reduce_tmp);
__syncwarp();
}
}
else {
int row_tmp = info_tmp.row_st;
float res_tmp = -99999;
for (int nnz_tmp = info_tmp.col_st + threadIdx.x; nnz_tmp < info_tmp.col_ed; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) {
atomicMax_float(&em[row_tmp * 1 + 0], res_tmp);
}
}
}
__global__ void k_16_fop_2(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* h, float* em, float* ho) {
int lane_id = threadIdx.x % WARP_SIZE;
int row_tmp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
if (row_tmp >= numnodes) return;
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);
}
}
__global__ void k_16_fop_3(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* ho, float* es) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *ep_n) return;
row_panel info_tmp = ep_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (info_tmp.col_st == -1){ 
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
using WarpReduce = cub::WarpReduce<float>;
__shared__ typename WarpReduce::TempStorage temp_storage[E_BLOCK_WARP_SIZE];
int row_tmp = info_tmp.row_st;
for (int col_iter = col_st; col_iter < col_ed; col_iter += blockDim.x) {
int nnz_tmp = col_iter + threadIdx.x;
int col_tmp = ColIdx[nnz_tmp];
int new_line_flag = 0; 
float res_tmp = 0;
if (nnz_tmp < col_ed) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
new_line_flag = (RowPtr[row_tmp] == nnz_tmp || lane_id == 0);
res_tmp = ho[nnz_tmp * 1 + 0];
}
float reduce_tmp = WarpReduce(temp_storage[local_wid]).HeadSegmentedReduce(res_tmp, new_line_flag, cub::Sum());
if (new_line_flag) atomicAdd(&es[row_tmp * 1 + 0], reduce_tmp);
__syncwarp();
}
}
else {
int row_tmp = info_tmp.row_st;
float res_tmp = 0;
for (int nnz_tmp = info_tmp.col_st + threadIdx.x; nnz_tmp < info_tmp.col_ed; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) {
atomicAdd(&es[row_tmp * 1 + 0], res_tmp);
}
}
}
__global__ void k_16_fop_4(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* ho, float* f, float* es, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
int row_tmp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
if (row_tmp >= numnodes) return;
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = RowPtr[row_tmp]; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
fo[row_tmp * featlen + k_tmp] = res_tmp;
}
}
void gat_kernel_16(int64_t info_addr, torch::Tensor RowPtr, torch::Tensor ColIdx, int64_t featlen, torch::Tensor f, torch::Tensor fo, torch::Tensor we, torch::Tensor lr, torch::Tensor e, torch::Tensor em, torch::Tensor es, torch::Tensor h, torch::Tensor ho) {
kg_info* info_ = (kg_info*)info_addr;
if (info_ == nullptr) {
  throw std::runtime_error("info_ is null in gat_kernel_16");
}
if (info_->rp_info == nullptr || info_->rp_n == nullptr || info_->ep_info == nullptr || info_->ep_n == nullptr || info_->ng_info == nullptr || info_->ng_n == nullptr) {
  throw std::runtime_error("Some preprocessing pointers are null in gat_kernel_16");
}
int numnodes = RowPtr.size(0) - 1;
int numedges = ColIdx.size(0);
int num_n_block = (numnodes + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
int num_ne_block = (numnodes + NE_BLOCK_WARP_SIZE - 1) / NE_BLOCK_WARP_SIZE;
int num_e_block = info_->ep_n_host;
int num_he_block = info_->rp_n_host;
int num_ngd_block = (info_->ng_n_host + NGD_BLOCK_WARP_SIZE - 1) / NGD_BLOCK_WARP_SIZE;
k_16_fop_0<<<num_ne_block, NE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, e.data_ptr<float>(), h.data_ptr<float>());
k_16_fop_1<<<num_e_block, E_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, h.data_ptr<float>(), lr.data_ptr<float>(), em.data_ptr<float>());
k_16_fop_2<<<num_ne_block, NE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, h.data_ptr<float>(), em.data_ptr<float>(), ho.data_ptr<float>());
k_16_fop_3<<<num_e_block, E_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, ho.data_ptr<float>(), es.data_ptr<float>());
k_16_fop_4<<<num_ne_block, NE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, ho.data_ptr<float>(), f.data_ptr<float>(), es.data_ptr<float>(), fo.data_ptr<float>());
}
__global__ void k17_fop0_dp0(int *RowPtr, int *ColIdx, int row_tmp, int featlen, float* e, float* h) {
int lane_id = threadIdx.x % WARP_SIZE;
int idx_st = RowPtr[row_tmp] + blockIdx.x * DP_NNZ_PER_BLOCK + threadIdx.x / WARP_SIZE * DP_NNZ_PER_WARP;
int idx_ed = idx_st + DP_NNZ_PER_WARP;
if (idx_ed > RowPtr[row_tmp + 1]) idx_ed = RowPtr[row_tmp + 1];
for (int nnz_tmp = idx_st + lane_id; nnz_tmp < idx_ed; nnz_tmp += WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];}
}
__global__ void k17_fop0_dp2(int *RowPtr, int *ColIdx, int row_tmp, int featlen, float* h, float* em, float* ho, float* es) {
int lane_id = threadIdx.x % WARP_SIZE;
int idx_st = RowPtr[row_tmp] + blockIdx.x * DP_NNZ_PER_BLOCK + threadIdx.x / WARP_SIZE * DP_NNZ_PER_WARP;
int idx_ed = idx_st + DP_NNZ_PER_WARP;
if (idx_ed > RowPtr[row_tmp + 1]) idx_ed = RowPtr[row_tmp + 1];
for (int nnz_tmp = idx_st + lane_id; nnz_tmp < idx_ed; nnz_tmp += WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);}
float res_tmp = 0;
for (int nnz_tmp = idx_st + lane_id; nnz_tmp < idx_ed; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) {
atomicAdd(&es[row_tmp * 1 + 0], res_tmp);
}
}
__global__ void k_17_fop_0(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* e, float* h, float* lr, float* em, float* ho, float* es) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *info_n) return;
row_panel info_tmp = info_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
int row_tmp_ne_st = info_tmp.row_st + local_wid;
int row_tmp_ne_ed = info_tmp.row_ed;
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
if (info_tmp.col_st == -1) {
{ for (int row_tmp = row_tmp_ne_st; row_tmp < row_tmp_ne_ed; row_tmp += HE_BLOCK_WARP_SIZE) {
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];
}
}
}
__syncthreads();
{ using WarpReduce = cub::WarpReduce<float>;
__shared__ typename WarpReduce::TempStorage temp_storage[HE_BLOCK_WARP_SIZE];
int row_tmp = info_tmp.row_st;
for (int col_iter = col_st; col_iter < col_ed; col_iter += blockDim.x) {
int nnz_tmp = col_iter + threadIdx.x;
int col_tmp = ColIdx[nnz_tmp];
int new_line_flag = 0; 
float res_tmp = 0;
if (nnz_tmp < col_ed) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
new_line_flag = (RowPtr[row_tmp] == nnz_tmp || lane_id == 0);
res_tmp = kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]);
}
else
res_tmp = -99999;
float reduce_tmp = WarpReduce(temp_storage[local_wid]).HeadSegmentedReduce(res_tmp, new_line_flag, cub::Max());
if (new_line_flag) atomicMax_float(&em[row_tmp * 1 + 0], reduce_tmp);
__syncwarp();
}
}
__syncthreads();
{ for (int row_tmp = row_tmp_ne_st; row_tmp < row_tmp_ne_ed; row_tmp += HE_BLOCK_WARP_SIZE) {
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);
}
}
}
__syncthreads();
{ using WarpReduce = cub::WarpReduce<float>;
__shared__ typename WarpReduce::TempStorage temp_storage[HE_BLOCK_WARP_SIZE];
int row_tmp = info_tmp.row_st;
for (int col_iter = col_st; col_iter < col_ed; col_iter += blockDim.x) {
int nnz_tmp = col_iter + threadIdx.x;
int col_tmp = ColIdx[nnz_tmp];
int new_line_flag = 0; 
float res_tmp = 0;
if (nnz_tmp < col_ed) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
new_line_flag = (RowPtr[row_tmp] == nnz_tmp || lane_id == 0);
res_tmp = ho[nnz_tmp * 1 + 0];
}
float reduce_tmp = WarpReduce(temp_storage[local_wid]).HeadSegmentedReduce(res_tmp, new_line_flag, cub::Sum());
if (new_line_flag) atomicAdd(&es[row_tmp * 1 + 0], reduce_tmp);
__syncwarp();
}
}
__syncthreads();
}
else {
int row_tmp = info_tmp.row_st;
int nnzs = RowPtr[row_tmp + 1] - RowPtr[row_tmp];
int dynamic_blocks = (nnzs + DP_NNZ_PER_BLOCK - 1) / DP_NNZ_PER_BLOCK;
if (threadIdx.x == 0) k17_fop0_dp0<<<dynamic_blocks, DP_BLOCK_SIZE>>>(RowPtr, ColIdx, row_tmp, featlen, e, h);
{
float res_tmp = -99999;
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) {
atomicMax_float(&em[row_tmp * 1 + 0], res_tmp);
}
}
if (threadIdx.x == 0) k17_fop0_dp2<<<dynamic_blocks, DP_BLOCK_SIZE>>>(RowPtr, ColIdx, row_tmp, featlen, h, em, ho, es);
}
}
__global__ void k_17_fop_1(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* ho, float* f, float* es, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
int row_tmp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
if (row_tmp >= numnodes) return;
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = RowPtr[row_tmp]; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
fo[row_tmp * featlen + k_tmp] = res_tmp;
}
}
void gat_kernel_17(int64_t info_addr, torch::Tensor RowPtr, torch::Tensor ColIdx, int64_t featlen, torch::Tensor f, torch::Tensor fo, torch::Tensor we, torch::Tensor lr, torch::Tensor e, torch::Tensor em, torch::Tensor es, torch::Tensor h, torch::Tensor ho) {
kg_info* info_ = (kg_info*)info_addr;
if (info_ == nullptr) {
  throw std::runtime_error("info_ is null in gat_kernel_17");
}
if (info_->rp_info == nullptr || info_->rp_n == nullptr || info_->ep_info == nullptr || info_->ep_n == nullptr || info_->ng_info == nullptr || info_->ng_n == nullptr) {
  throw std::runtime_error("Some preprocessing pointers are null in gat_kernel_17");
}
int numnodes = RowPtr.size(0) - 1;
int numedges = ColIdx.size(0);
int num_n_block = (numnodes + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
int num_ne_block = (numnodes + NE_BLOCK_WARP_SIZE - 1) / NE_BLOCK_WARP_SIZE;
int num_e_block = info_->ep_n_host;
int num_he_block = info_->rp_n_host;
int num_ngd_block = (info_->ng_n_host + NGD_BLOCK_WARP_SIZE - 1) / NGD_BLOCK_WARP_SIZE;
k_17_fop_0<<<num_he_block, HE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, e.data_ptr<float>(), h.data_ptr<float>(), lr.data_ptr<float>(), em.data_ptr<float>(), ho.data_ptr<float>(), es.data_ptr<float>());
k_17_fop_1<<<num_ne_block, NE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, ho.data_ptr<float>(), f.data_ptr<float>(), es.data_ptr<float>(), fo.data_ptr<float>());
}
__global__ void k18_fop0_dp4(int *RowPtr, int *ColIdx, int row_tmp, int featlen, float* ho, float* f, float* es, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
int idx_st = RowPtr[row_tmp] + blockIdx.x * DP_NNZ_PER_BLOCK + threadIdx.x / WARP_SIZE * DP_NNZ_PER_WARP;
int idx_ed = idx_st + DP_NNZ_PER_WARP;
if (idx_ed > RowPtr[row_tmp + 1]) idx_ed = RowPtr[row_tmp + 1];
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = idx_st; nnz_tmp < idx_ed; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];
}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
atomicAdd(&fo[row_tmp * featlen + k_tmp], res_tmp);
}
}
__global__ void k_18_fop_0(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* e, float* h, float* lr, float* em, float* ho, float* es, float* f, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *info_n) return;
row_panel info_tmp = info_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
int row_tmp_ne_st = info_tmp.row_st + local_wid;
int row_tmp_ne_ed = info_tmp.row_ed;
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
if (info_tmp.col_st == -1) {
{ for (int row_tmp = row_tmp_ne_st; row_tmp < row_tmp_ne_ed; row_tmp += HE_BLOCK_WARP_SIZE) {
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];
}
}
}
__syncthreads();
{ using WarpReduce = cub::WarpReduce<float>;
__shared__ typename WarpReduce::TempStorage temp_storage[HE_BLOCK_WARP_SIZE];
int row_tmp = info_tmp.row_st;
for (int col_iter = col_st; col_iter < col_ed; col_iter += blockDim.x) {
int nnz_tmp = col_iter + threadIdx.x;
int col_tmp = ColIdx[nnz_tmp];
int new_line_flag = 0; 
float res_tmp = 0;
if (nnz_tmp < col_ed) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
new_line_flag = (RowPtr[row_tmp] == nnz_tmp || lane_id == 0);
res_tmp = kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]);
}
else
res_tmp = -99999;
float reduce_tmp = WarpReduce(temp_storage[local_wid]).HeadSegmentedReduce(res_tmp, new_line_flag, cub::Max());
if (new_line_flag) atomicMax_float(&em[row_tmp * 1 + 0], reduce_tmp);
__syncwarp();
}
}
__syncthreads();
{ for (int row_tmp = row_tmp_ne_st; row_tmp < row_tmp_ne_ed; row_tmp += HE_BLOCK_WARP_SIZE) {
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);
}
}
}
__syncthreads();
{ using WarpReduce = cub::WarpReduce<float>;
__shared__ typename WarpReduce::TempStorage temp_storage[HE_BLOCK_WARP_SIZE];
int row_tmp = info_tmp.row_st;
for (int col_iter = col_st; col_iter < col_ed; col_iter += blockDim.x) {
int nnz_tmp = col_iter + threadIdx.x;
int col_tmp = ColIdx[nnz_tmp];
int new_line_flag = 0; 
float res_tmp = 0;
if (nnz_tmp < col_ed) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
new_line_flag = (RowPtr[row_tmp] == nnz_tmp || lane_id == 0);
res_tmp = ho[nnz_tmp * 1 + 0];
}
float reduce_tmp = WarpReduce(temp_storage[local_wid]).HeadSegmentedReduce(res_tmp, new_line_flag, cub::Sum());
if (new_line_flag) atomicAdd(&es[row_tmp * 1 + 0], reduce_tmp);
__syncwarp();
}
}
__syncthreads();
{ for (int row_tmp = row_tmp_ne_st; row_tmp < row_tmp_ne_ed; row_tmp += HE_BLOCK_WARP_SIZE) {
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = RowPtr[row_tmp]; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];
}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
fo[row_tmp * featlen + k_tmp] = res_tmp;
}
}
}
__syncthreads();
}
else {
int row_tmp = info_tmp.row_st;
int nnzs = RowPtr[row_tmp + 1] - RowPtr[row_tmp];
int dynamic_blocks = (nnzs + DP_NNZ_PER_BLOCK - 1) / DP_NNZ_PER_BLOCK;
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];}__syncthreads();
{
float res_tmp = -99999;
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) {
atomicMax_float(&em[row_tmp * 1 + 0], res_tmp);
}
}__syncthreads();
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);}__syncthreads();
{
float res_tmp = 0;
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) {
atomicAdd(&es[row_tmp * 1 + 0], res_tmp);
}
}__syncthreads();
if (threadIdx.x == 0) k18_fop0_dp4<<<dynamic_blocks, DP_BLOCK_SIZE>>>(RowPtr, ColIdx, row_tmp, featlen, ho, f, es, fo);
}
}
void gat_kernel_18(int64_t info_addr, torch::Tensor RowPtr, torch::Tensor ColIdx, int64_t featlen, torch::Tensor f, torch::Tensor fo, torch::Tensor we, torch::Tensor lr, torch::Tensor e, torch::Tensor em, torch::Tensor es, torch::Tensor h, torch::Tensor ho) {
kg_info* info_ = (kg_info*)info_addr;
if (info_ == nullptr) {
  throw std::runtime_error("info_ is null in gat_kernel_18");
}
if (info_->rp_info == nullptr || info_->rp_n == nullptr || info_->ep_info == nullptr || info_->ep_n == nullptr || info_->ng_info == nullptr || info_->ng_n == nullptr) {
  throw std::runtime_error("Some preprocessing pointers are null in gat_kernel_18");
}
int numnodes = RowPtr.size(0) - 1;
int numedges = ColIdx.size(0);
int num_n_block = (numnodes + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
int num_ne_block = (numnodes + NE_BLOCK_WARP_SIZE - 1) / NE_BLOCK_WARP_SIZE;
int num_e_block = info_->ep_n_host;
int num_he_block = info_->rp_n_host;
int num_ngd_block = (info_->ng_n_host + NGD_BLOCK_WARP_SIZE - 1) / NGD_BLOCK_WARP_SIZE;
k_18_fop_0<<<num_he_block, HE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, e.data_ptr<float>(), h.data_ptr<float>(), lr.data_ptr<float>(), em.data_ptr<float>(), ho.data_ptr<float>(), es.data_ptr<float>(), f.data_ptr<float>(), fo.data_ptr<float>());
}
__global__ void k_19_fop_0(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* e, float* h) {
int lane_id = threadIdx.x % WARP_SIZE;
int row_tmp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
if (row_tmp >= numnodes) return;
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];
}
}
__global__ void k_19_fop_1(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* h, float* lr, float* em) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *ep_n) return;
row_panel info_tmp = ep_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (info_tmp.col_st == -1){ 
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
using WarpReduce = cub::WarpReduce<float>;
__shared__ typename WarpReduce::TempStorage temp_storage[E_BLOCK_WARP_SIZE];
int row_tmp = info_tmp.row_st;
for (int col_iter = col_st; col_iter < col_ed; col_iter += blockDim.x) {
int nnz_tmp = col_iter + threadIdx.x;
int col_tmp = ColIdx[nnz_tmp];
int new_line_flag = 0; 
float res_tmp = 0;
if (nnz_tmp < col_ed) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
new_line_flag = (RowPtr[row_tmp] == nnz_tmp || lane_id == 0);
res_tmp = kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]);
}
else
res_tmp = -99999;
float reduce_tmp = WarpReduce(temp_storage[local_wid]).HeadSegmentedReduce(res_tmp, new_line_flag, cub::Max());
if (new_line_flag) atomicMax_float(&em[row_tmp * 1 + 0], reduce_tmp);
__syncwarp();
}
}
else {
int row_tmp = info_tmp.row_st;
float res_tmp = -99999;
for (int nnz_tmp = info_tmp.col_st + threadIdx.x; nnz_tmp < info_tmp.col_ed; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) {
atomicMax_float(&em[row_tmp * 1 + 0], res_tmp);
}
}
}
__global__ void k_19_fop_2(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* h, float* em, float* ho) {
int lane_id = threadIdx.x % WARP_SIZE;
int row_tmp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
if (row_tmp >= numnodes) return;
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);
}
}
__global__ void k_19_fop_3(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* ho, float* es) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *ep_n) return;
row_panel info_tmp = ep_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (info_tmp.col_st == -1){ 
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
using WarpReduce = cub::WarpReduce<float>;
__shared__ typename WarpReduce::TempStorage temp_storage[E_BLOCK_WARP_SIZE];
int row_tmp = info_tmp.row_st;
for (int col_iter = col_st; col_iter < col_ed; col_iter += blockDim.x) {
int nnz_tmp = col_iter + threadIdx.x;
int col_tmp = ColIdx[nnz_tmp];
int new_line_flag = 0; 
float res_tmp = 0;
if (nnz_tmp < col_ed) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
new_line_flag = (RowPtr[row_tmp] == nnz_tmp || lane_id == 0);
res_tmp = ho[nnz_tmp * 1 + 0];
}
float reduce_tmp = WarpReduce(temp_storage[local_wid]).HeadSegmentedReduce(res_tmp, new_line_flag, cub::Sum());
if (new_line_flag) atomicAdd(&es[row_tmp * 1 + 0], reduce_tmp);
__syncwarp();
}
}
else {
int row_tmp = info_tmp.row_st;
float res_tmp = 0;
for (int nnz_tmp = info_tmp.col_st + threadIdx.x; nnz_tmp < info_tmp.col_ed; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) {
atomicAdd(&es[row_tmp * 1 + 0], res_tmp);
}
}
}
__global__ void k_19_fop_4(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* ho, float* f, float* es, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *ep_n) return;
row_panel info_tmp = ep_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (info_tmp.col_st == -1){ 
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
int nnz_per_warp = (nnzs + E_BLOCK_WARP_SIZE - 1) / E_BLOCK_WARP_SIZE;
int warp_col_st = col_st + nnz_per_warp * local_wid;
int warp_col_ed = (warp_col_st + nnz_per_warp < col_ed)? warp_col_st + nnz_per_warp: col_ed;
int warp_row_st = info_tmp.row_st, warp_row_ed;
while (warp_row_st < info_tmp.row_ed - 1 && warp_col_st > RowPtr[warp_row_st + 1]) {
warp_row_st++;
}
warp_row_ed = warp_row_st;
while (warp_row_ed < info_tmp.row_ed && warp_col_ed > RowPtr[warp_row_ed]) {
warp_row_ed++;
}
if (warp_col_ed <= col_ed) {
for (int row_tmp = warp_row_st; row_tmp < warp_row_ed; row_tmp++) {
int current_col_st = RowPtr[row_tmp];
int current_col_ed = RowPtr[row_tmp + 1];
if (row_tmp == warp_row_st) current_col_st = warp_col_st;
if (row_tmp == warp_row_ed - 1) current_col_ed = warp_col_ed;
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = current_col_st; nnz_tmp < current_col_ed; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
atomicAdd(&fo[row_tmp * featlen + k_tmp], res_tmp);
}}}
}
else {
int row_tmp = info_tmp.row_st;
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = info_tmp.col_st + local_wid; nnz_tmp < info_tmp.col_ed; nnz_tmp+=HE_BLOCK_WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];
}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
atomicAdd(&fo[row_tmp * featlen + k_tmp], res_tmp);
}
}
}
void gat_kernel_19(int64_t info_addr, torch::Tensor RowPtr, torch::Tensor ColIdx, int64_t featlen, torch::Tensor f, torch::Tensor fo, torch::Tensor we, torch::Tensor lr, torch::Tensor e, torch::Tensor em, torch::Tensor es, torch::Tensor h, torch::Tensor ho) {
kg_info* info_ = (kg_info*)info_addr;
if (info_ == nullptr) {
  throw std::runtime_error("info_ is null in gat_kernel_19");
}
if (info_->rp_info == nullptr || info_->rp_n == nullptr || info_->ep_info == nullptr || info_->ep_n == nullptr || info_->ng_info == nullptr || info_->ng_n == nullptr) {
  throw std::runtime_error("Some preprocessing pointers are null in gat_kernel_19");
}
int numnodes = RowPtr.size(0) - 1;
int numedges = ColIdx.size(0);
int num_n_block = (numnodes + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
int num_ne_block = (numnodes + NE_BLOCK_WARP_SIZE - 1) / NE_BLOCK_WARP_SIZE;
int num_e_block = info_->ep_n_host;
int num_he_block = info_->rp_n_host;
int num_ngd_block = (info_->ng_n_host + NGD_BLOCK_WARP_SIZE - 1) / NGD_BLOCK_WARP_SIZE;
k_19_fop_0<<<num_ne_block, NE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, e.data_ptr<float>(), h.data_ptr<float>());
k_19_fop_1<<<num_e_block, E_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, h.data_ptr<float>(), lr.data_ptr<float>(), em.data_ptr<float>());
k_19_fop_2<<<num_ne_block, NE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, h.data_ptr<float>(), em.data_ptr<float>(), ho.data_ptr<float>());
k_19_fop_3<<<num_e_block, E_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, ho.data_ptr<float>(), es.data_ptr<float>());
k_19_fop_4<<<num_e_block, E_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, ho.data_ptr<float>(), f.data_ptr<float>(), es.data_ptr<float>(), fo.data_ptr<float>());
}
__global__ void k20_fop0_dp0(int *RowPtr, int *ColIdx, int row_tmp, int featlen, float* e, float* h) {
int lane_id = threadIdx.x % WARP_SIZE;
int idx_st = RowPtr[row_tmp] + blockIdx.x * DP_NNZ_PER_BLOCK + threadIdx.x / WARP_SIZE * DP_NNZ_PER_WARP;
int idx_ed = idx_st + DP_NNZ_PER_WARP;
if (idx_ed > RowPtr[row_tmp + 1]) idx_ed = RowPtr[row_tmp + 1];
for (int nnz_tmp = idx_st + lane_id; nnz_tmp < idx_ed; nnz_tmp += WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];}
}
__global__ void k20_fop0_dp2(int *RowPtr, int *ColIdx, int row_tmp, int featlen, float* h, float* em, float* ho, float* es) {
int lane_id = threadIdx.x % WARP_SIZE;
int idx_st = RowPtr[row_tmp] + blockIdx.x * DP_NNZ_PER_BLOCK + threadIdx.x / WARP_SIZE * DP_NNZ_PER_WARP;
int idx_ed = idx_st + DP_NNZ_PER_WARP;
if (idx_ed > RowPtr[row_tmp + 1]) idx_ed = RowPtr[row_tmp + 1];
for (int nnz_tmp = idx_st + lane_id; nnz_tmp < idx_ed; nnz_tmp += WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);}
float res_tmp = 0;
for (int nnz_tmp = idx_st + lane_id; nnz_tmp < idx_ed; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) {
atomicAdd(&es[row_tmp * 1 + 0], res_tmp);
}
}
__global__ void k_20_fop_0(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* e, float* h, float* lr, float* em, float* ho, float* es) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *info_n) return;
row_panel info_tmp = info_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
int row_tmp_ne_st = info_tmp.row_st + local_wid;
int row_tmp_ne_ed = info_tmp.row_ed;
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
if (info_tmp.col_st == -1) {
{ for (int row_tmp = row_tmp_ne_st; row_tmp < row_tmp_ne_ed; row_tmp += HE_BLOCK_WARP_SIZE) {
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];
}
}
}
__syncthreads();
{ using WarpReduce = cub::WarpReduce<float>;
__shared__ typename WarpReduce::TempStorage temp_storage[HE_BLOCK_WARP_SIZE];
int row_tmp = info_tmp.row_st;
for (int col_iter = col_st; col_iter < col_ed; col_iter += blockDim.x) {
int nnz_tmp = col_iter + threadIdx.x;
int col_tmp = ColIdx[nnz_tmp];
int new_line_flag = 0; 
float res_tmp = 0;
if (nnz_tmp < col_ed) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
new_line_flag = (RowPtr[row_tmp] == nnz_tmp || lane_id == 0);
res_tmp = kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]);
}
else
res_tmp = -99999;
float reduce_tmp = WarpReduce(temp_storage[local_wid]).HeadSegmentedReduce(res_tmp, new_line_flag, cub::Max());
if (new_line_flag) atomicMax_float(&em[row_tmp * 1 + 0], reduce_tmp);
__syncwarp();
}
}
__syncthreads();
{ for (int row_tmp = row_tmp_ne_st; row_tmp < row_tmp_ne_ed; row_tmp += HE_BLOCK_WARP_SIZE) {
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);
}
}
}
__syncthreads();
{ using WarpReduce = cub::WarpReduce<float>;
__shared__ typename WarpReduce::TempStorage temp_storage[HE_BLOCK_WARP_SIZE];
int row_tmp = info_tmp.row_st;
for (int col_iter = col_st; col_iter < col_ed; col_iter += blockDim.x) {
int nnz_tmp = col_iter + threadIdx.x;
int col_tmp = ColIdx[nnz_tmp];
int new_line_flag = 0; 
float res_tmp = 0;
if (nnz_tmp < col_ed) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
new_line_flag = (RowPtr[row_tmp] == nnz_tmp || lane_id == 0);
res_tmp = ho[nnz_tmp * 1 + 0];
}
float reduce_tmp = WarpReduce(temp_storage[local_wid]).HeadSegmentedReduce(res_tmp, new_line_flag, cub::Sum());
if (new_line_flag) atomicAdd(&es[row_tmp * 1 + 0], reduce_tmp);
__syncwarp();
}
}
__syncthreads();
}
else {
int row_tmp = info_tmp.row_st;
int nnzs = RowPtr[row_tmp + 1] - RowPtr[row_tmp];
int dynamic_blocks = (nnzs + DP_NNZ_PER_BLOCK - 1) / DP_NNZ_PER_BLOCK;
if (threadIdx.x == 0) k20_fop0_dp0<<<dynamic_blocks, DP_BLOCK_SIZE>>>(RowPtr, ColIdx, row_tmp, featlen, e, h);
{
float res_tmp = -99999;
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) {
atomicMax_float(&em[row_tmp * 1 + 0], res_tmp);
}
}
if (threadIdx.x == 0) k20_fop0_dp2<<<dynamic_blocks, DP_BLOCK_SIZE>>>(RowPtr, ColIdx, row_tmp, featlen, h, em, ho, es);
}
}
__global__ void k_20_fop_1(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* ho, float* f, float* es, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *ep_n) return;
row_panel info_tmp = ep_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (info_tmp.col_st == -1){ 
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
int nnz_per_warp = (nnzs + E_BLOCK_WARP_SIZE - 1) / E_BLOCK_WARP_SIZE;
int warp_col_st = col_st + nnz_per_warp * local_wid;
int warp_col_ed = (warp_col_st + nnz_per_warp < col_ed)? warp_col_st + nnz_per_warp: col_ed;
int warp_row_st = info_tmp.row_st, warp_row_ed;
while (warp_row_st < info_tmp.row_ed - 1 && warp_col_st > RowPtr[warp_row_st + 1]) {
warp_row_st++;
}
warp_row_ed = warp_row_st;
while (warp_row_ed < info_tmp.row_ed && warp_col_ed > RowPtr[warp_row_ed]) {
warp_row_ed++;
}
if (warp_col_ed <= col_ed) {
for (int row_tmp = warp_row_st; row_tmp < warp_row_ed; row_tmp++) {
int current_col_st = RowPtr[row_tmp];
int current_col_ed = RowPtr[row_tmp + 1];
if (row_tmp == warp_row_st) current_col_st = warp_col_st;
if (row_tmp == warp_row_ed - 1) current_col_ed = warp_col_ed;
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = current_col_st; nnz_tmp < current_col_ed; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
atomicAdd(&fo[row_tmp * featlen + k_tmp], res_tmp);
}}}
}
else {
int row_tmp = info_tmp.row_st;
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = info_tmp.col_st + local_wid; nnz_tmp < info_tmp.col_ed; nnz_tmp+=HE_BLOCK_WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];
}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
atomicAdd(&fo[row_tmp * featlen + k_tmp], res_tmp);
}
}
}
void gat_kernel_20(int64_t info_addr, torch::Tensor RowPtr, torch::Tensor ColIdx, int64_t featlen, torch::Tensor f, torch::Tensor fo, torch::Tensor we, torch::Tensor lr, torch::Tensor e, torch::Tensor em, torch::Tensor es, torch::Tensor h, torch::Tensor ho) {
kg_info* info_ = (kg_info*)info_addr;
if (info_ == nullptr) {
  throw std::runtime_error("info_ is null in gat_kernel_20");
}
if (info_->rp_info == nullptr || info_->rp_n == nullptr || info_->ep_info == nullptr || info_->ep_n == nullptr || info_->ng_info == nullptr || info_->ng_n == nullptr) {
  throw std::runtime_error("Some preprocessing pointers are null in gat_kernel_20");
}
int numnodes = RowPtr.size(0) - 1;
int numedges = ColIdx.size(0);
int num_n_block = (numnodes + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
int num_ne_block = (numnodes + NE_BLOCK_WARP_SIZE - 1) / NE_BLOCK_WARP_SIZE;
int num_e_block = info_->ep_n_host;
int num_he_block = info_->rp_n_host;
int num_ngd_block = (info_->ng_n_host + NGD_BLOCK_WARP_SIZE - 1) / NGD_BLOCK_WARP_SIZE;
k_20_fop_0<<<num_he_block, HE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, e.data_ptr<float>(), h.data_ptr<float>(), lr.data_ptr<float>(), em.data_ptr<float>(), ho.data_ptr<float>(), es.data_ptr<float>());
k_20_fop_1<<<num_e_block, E_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, ho.data_ptr<float>(), f.data_ptr<float>(), es.data_ptr<float>(), fo.data_ptr<float>());
}
__global__ void k21_fop0_dp4(int *RowPtr, int *ColIdx, int row_tmp, int featlen, float* ho, float* f, float* es, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
int idx_st = RowPtr[row_tmp] + blockIdx.x * DP_NNZ_PER_BLOCK + threadIdx.x / WARP_SIZE * DP_NNZ_PER_WARP;
int idx_ed = idx_st + DP_NNZ_PER_WARP;
if (idx_ed > RowPtr[row_tmp + 1]) idx_ed = RowPtr[row_tmp + 1];
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = idx_st; nnz_tmp < idx_ed; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];
}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
atomicAdd(&fo[row_tmp * featlen + k_tmp], res_tmp);
}
}
__global__ void k_21_fop_0(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* e, float* h, float* lr, float* em, float* ho, float* es, float* f, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *info_n) return;
row_panel info_tmp = info_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
int row_tmp_ne_st = info_tmp.row_st + local_wid;
int row_tmp_ne_ed = info_tmp.row_ed;
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
int nnz_per_warp = (nnzs + HE_BLOCK_WARP_SIZE - 1) / HE_BLOCK_WARP_SIZE;
int warp_col_st = col_st + nnz_per_warp * local_wid;
int warp_col_ed = (warp_col_st + nnz_per_warp < col_ed)? warp_col_st + nnz_per_warp: col_ed;
int warp_row_st = info_tmp.row_st, warp_row_ed;
while (warp_row_st < info_tmp.row_ed - 1 && warp_col_st > RowPtr[warp_row_st + 1]) {
warp_row_st++;
}
warp_row_ed = warp_row_st;
while (warp_row_ed < info_tmp.row_ed && warp_col_ed > RowPtr[warp_row_ed]) {
warp_row_ed++;
}
if (info_tmp.col_st == -1) {
{ for (int row_tmp = row_tmp_ne_st; row_tmp < row_tmp_ne_ed; row_tmp += HE_BLOCK_WARP_SIZE) {
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];
}
}
}
__syncthreads();
{ using WarpReduce = cub::WarpReduce<float>;
__shared__ typename WarpReduce::TempStorage temp_storage[HE_BLOCK_WARP_SIZE];
int row_tmp = info_tmp.row_st;
for (int col_iter = col_st; col_iter < col_ed; col_iter += blockDim.x) {
int nnz_tmp = col_iter + threadIdx.x;
int col_tmp = ColIdx[nnz_tmp];
int new_line_flag = 0; 
float res_tmp = 0;
if (nnz_tmp < col_ed) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
new_line_flag = (RowPtr[row_tmp] == nnz_tmp || lane_id == 0);
res_tmp = kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]);
}
else
res_tmp = -99999;
float reduce_tmp = WarpReduce(temp_storage[local_wid]).HeadSegmentedReduce(res_tmp, new_line_flag, cub::Max());
if (new_line_flag) atomicMax_float(&em[row_tmp * 1 + 0], reduce_tmp);
__syncwarp();
}
}
__syncthreads();
{ for (int row_tmp = row_tmp_ne_st; row_tmp < row_tmp_ne_ed; row_tmp += HE_BLOCK_WARP_SIZE) {
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);
}
}
}
__syncthreads();
{ using WarpReduce = cub::WarpReduce<float>;
__shared__ typename WarpReduce::TempStorage temp_storage[HE_BLOCK_WARP_SIZE];
int row_tmp = info_tmp.row_st;
for (int col_iter = col_st; col_iter < col_ed; col_iter += blockDim.x) {
int nnz_tmp = col_iter + threadIdx.x;
int col_tmp = ColIdx[nnz_tmp];
int new_line_flag = 0; 
float res_tmp = 0;
if (nnz_tmp < col_ed) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
new_line_flag = (RowPtr[row_tmp] == nnz_tmp || lane_id == 0);
res_tmp = ho[nnz_tmp * 1 + 0];
}
float reduce_tmp = WarpReduce(temp_storage[local_wid]).HeadSegmentedReduce(res_tmp, new_line_flag, cub::Sum());
if (new_line_flag) atomicAdd(&es[row_tmp * 1 + 0], reduce_tmp);
__syncwarp();
}
}
__syncthreads();
{ if (warp_col_ed <= col_ed) {
for (int row_tmp = warp_row_st; row_tmp < warp_row_ed; row_tmp++) {
int current_col_st = RowPtr[row_tmp];
int current_col_ed = RowPtr[row_tmp + 1];
if (row_tmp == warp_row_st) current_col_st = warp_col_st;
if (row_tmp == warp_row_ed - 1) current_col_ed = warp_col_ed;
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = current_col_st; nnz_tmp < current_col_ed; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];
}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
atomicAdd(&fo[row_tmp * featlen + k_tmp], res_tmp);
}}}
}
__syncthreads();
}
else {
int row_tmp = info_tmp.row_st;
int nnzs = RowPtr[row_tmp + 1] - RowPtr[row_tmp];
int dynamic_blocks = (nnzs + DP_NNZ_PER_BLOCK - 1) / DP_NNZ_PER_BLOCK;
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];}__syncthreads();
{
float res_tmp = -99999;
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) {
atomicMax_float(&em[row_tmp * 1 + 0], res_tmp);
}
}__syncthreads();
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);}__syncthreads();
{
float res_tmp = 0;
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) {
atomicAdd(&es[row_tmp * 1 + 0], res_tmp);
}
}__syncthreads();
if (threadIdx.x == 0) k21_fop0_dp4<<<dynamic_blocks, DP_BLOCK_SIZE>>>(RowPtr, ColIdx, row_tmp, featlen, ho, f, es, fo);
}
}
void gat_kernel_21(int64_t info_addr, torch::Tensor RowPtr, torch::Tensor ColIdx, int64_t featlen, torch::Tensor f, torch::Tensor fo, torch::Tensor we, torch::Tensor lr, torch::Tensor e, torch::Tensor em, torch::Tensor es, torch::Tensor h, torch::Tensor ho) {
kg_info* info_ = (kg_info*)info_addr;
if (info_ == nullptr) {
  throw std::runtime_error("info_ is null in gat_kernel_21");
}
if (info_->rp_info == nullptr || info_->rp_n == nullptr || info_->ep_info == nullptr || info_->ep_n == nullptr || info_->ng_info == nullptr || info_->ng_n == nullptr) {
  throw std::runtime_error("Some preprocessing pointers are null in gat_kernel_21");
}
int numnodes = RowPtr.size(0) - 1;
int numedges = ColIdx.size(0);
int num_n_block = (numnodes + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
int num_ne_block = (numnodes + NE_BLOCK_WARP_SIZE - 1) / NE_BLOCK_WARP_SIZE;
int num_e_block = info_->ep_n_host;
int num_he_block = info_->rp_n_host;
int num_ngd_block = (info_->ng_n_host + NGD_BLOCK_WARP_SIZE - 1) / NGD_BLOCK_WARP_SIZE;
k_21_fop_0<<<num_he_block, HE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, e.data_ptr<float>(), h.data_ptr<float>(), lr.data_ptr<float>(), em.data_ptr<float>(), ho.data_ptr<float>(), es.data_ptr<float>(), f.data_ptr<float>(), fo.data_ptr<float>());
}
__global__ void k_22_fop_0(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* e, float* h) {
int lane_id = threadIdx.x % WARP_SIZE;
int row_tmp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
if (row_tmp >= numnodes) return;
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];
}
}
__global__ void k_22_fop_1(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* h, float* lr, float* em) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *ep_n) return;
row_panel info_tmp = ep_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (info_tmp.col_st == -1){ 
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
using WarpReduce = cub::WarpReduce<float>;
__shared__ typename WarpReduce::TempStorage temp_storage[E_BLOCK_WARP_SIZE];
int row_tmp = info_tmp.row_st;
for (int col_iter = col_st; col_iter < col_ed; col_iter += blockDim.x) {
int nnz_tmp = col_iter + threadIdx.x;
int col_tmp = ColIdx[nnz_tmp];
int new_line_flag = 0; 
float res_tmp = 0;
if (nnz_tmp < col_ed) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
new_line_flag = (RowPtr[row_tmp] == nnz_tmp || lane_id == 0);
res_tmp = kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]);
}
else
res_tmp = -99999;
float reduce_tmp = WarpReduce(temp_storage[local_wid]).HeadSegmentedReduce(res_tmp, new_line_flag, cub::Max());
if (new_line_flag) atomicMax_float(&em[row_tmp * 1 + 0], reduce_tmp);
__syncwarp();
}
}
else {
int row_tmp = info_tmp.row_st;
float res_tmp = -99999;
for (int nnz_tmp = info_tmp.col_st + threadIdx.x; nnz_tmp < info_tmp.col_ed; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) {
atomicMax_float(&em[row_tmp * 1 + 0], res_tmp);
}
}
}
__global__ void k_22_fop_2(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* h, float* em, float* ho) {
int lane_id = threadIdx.x % WARP_SIZE;
int row_tmp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
if (row_tmp >= numnodes) return;
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);
}
}
__global__ void k_22_fop_3(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* ho, float* es) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *ep_n) return;
row_panel info_tmp = ep_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (info_tmp.col_st == -1){ 
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
using WarpReduce = cub::WarpReduce<float>;
__shared__ typename WarpReduce::TempStorage temp_storage[E_BLOCK_WARP_SIZE];
int row_tmp = info_tmp.row_st;
for (int col_iter = col_st; col_iter < col_ed; col_iter += blockDim.x) {
int nnz_tmp = col_iter + threadIdx.x;
int col_tmp = ColIdx[nnz_tmp];
int new_line_flag = 0; 
float res_tmp = 0;
if (nnz_tmp < col_ed) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
new_line_flag = (RowPtr[row_tmp] == nnz_tmp || lane_id == 0);
res_tmp = ho[nnz_tmp * 1 + 0];
}
float reduce_tmp = WarpReduce(temp_storage[local_wid]).HeadSegmentedReduce(res_tmp, new_line_flag, cub::Sum());
if (new_line_flag) atomicAdd(&es[row_tmp * 1 + 0], reduce_tmp);
__syncwarp();
}
}
else {
int row_tmp = info_tmp.row_st;
float res_tmp = 0;
for (int nnz_tmp = info_tmp.col_st + threadIdx.x; nnz_tmp < info_tmp.col_ed; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) {
atomicAdd(&es[row_tmp * 1 + 0], res_tmp);
}
}
}
__global__ void k_22_fop_4(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* ho, float* f, float* es, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (global_wid >= *ng_n) return;
neighbor_group ng_tmp = ng_list[global_wid];
int row_tmp = ng_tmp.row_st;
int col_st = ng_tmp.col_st;
int col_ed = ng_tmp.col_st + NG_SIZE;
if (col_ed >= RowPtr[row_tmp + 1]) col_ed = RowPtr[row_tmp + 1];
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = col_st; nnz_tmp < col_ed; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
atomicAdd(&fo[row_tmp * featlen + k_tmp], res_tmp);
}
}
void gat_kernel_22(int64_t info_addr, torch::Tensor RowPtr, torch::Tensor ColIdx, int64_t featlen, torch::Tensor f, torch::Tensor fo, torch::Tensor we, torch::Tensor lr, torch::Tensor e, torch::Tensor em, torch::Tensor es, torch::Tensor h, torch::Tensor ho) {
kg_info* info_ = (kg_info*)info_addr;
if (info_ == nullptr) {
  throw std::runtime_error("info_ is null in gat_kernel_22");
}
if (info_->rp_info == nullptr || info_->rp_n == nullptr || info_->ep_info == nullptr || info_->ep_n == nullptr || info_->ng_info == nullptr || info_->ng_n == nullptr) {
  throw std::runtime_error("Some preprocessing pointers are null in gat_kernel_22");
}
int numnodes = RowPtr.size(0) - 1;
int numedges = ColIdx.size(0);
int num_n_block = (numnodes + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
int num_ne_block = (numnodes + NE_BLOCK_WARP_SIZE - 1) / NE_BLOCK_WARP_SIZE;
int num_e_block = info_->ep_n_host;
int num_he_block = info_->rp_n_host;
int num_ngd_block = (info_->ng_n_host + NGD_BLOCK_WARP_SIZE - 1) / NGD_BLOCK_WARP_SIZE;
k_22_fop_0<<<num_ne_block, NE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, e.data_ptr<float>(), h.data_ptr<float>());
k_22_fop_1<<<num_e_block, E_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, h.data_ptr<float>(), lr.data_ptr<float>(), em.data_ptr<float>());
k_22_fop_2<<<num_ne_block, NE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, h.data_ptr<float>(), em.data_ptr<float>(), ho.data_ptr<float>());
k_22_fop_3<<<num_e_block, E_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, ho.data_ptr<float>(), es.data_ptr<float>());
k_22_fop_4<<<num_ngd_block, NGD_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, ho.data_ptr<float>(), f.data_ptr<float>(), es.data_ptr<float>(), fo.data_ptr<float>());
}
__global__ void k23_fop0_dp0(int *RowPtr, int *ColIdx, int row_tmp, int featlen, float* e, float* h) {
int lane_id = threadIdx.x % WARP_SIZE;
int idx_st = RowPtr[row_tmp] + blockIdx.x * DP_NNZ_PER_BLOCK + threadIdx.x / WARP_SIZE * DP_NNZ_PER_WARP;
int idx_ed = idx_st + DP_NNZ_PER_WARP;
if (idx_ed > RowPtr[row_tmp + 1]) idx_ed = RowPtr[row_tmp + 1];
for (int nnz_tmp = idx_st + lane_id; nnz_tmp < idx_ed; nnz_tmp += WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];}
}
__global__ void k23_fop0_dp2(int *RowPtr, int *ColIdx, int row_tmp, int featlen, float* h, float* em, float* ho, float* es) {
int lane_id = threadIdx.x % WARP_SIZE;
int idx_st = RowPtr[row_tmp] + blockIdx.x * DP_NNZ_PER_BLOCK + threadIdx.x / WARP_SIZE * DP_NNZ_PER_WARP;
int idx_ed = idx_st + DP_NNZ_PER_WARP;
if (idx_ed > RowPtr[row_tmp + 1]) idx_ed = RowPtr[row_tmp + 1];
for (int nnz_tmp = idx_st + lane_id; nnz_tmp < idx_ed; nnz_tmp += WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);}
float res_tmp = 0;
for (int nnz_tmp = idx_st + lane_id; nnz_tmp < idx_ed; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) {
atomicAdd(&es[row_tmp * 1 + 0], res_tmp);
}
}
__global__ void k_23_fop_0(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* e, float* h, float* lr, float* em, float* ho, float* es) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *info_n) return;
row_panel info_tmp = info_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
int row_tmp_ne_st = info_tmp.row_st + local_wid;
int row_tmp_ne_ed = info_tmp.row_ed;
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
if (info_tmp.col_st == -1) {
{ for (int row_tmp = row_tmp_ne_st; row_tmp < row_tmp_ne_ed; row_tmp += HE_BLOCK_WARP_SIZE) {
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];
}
}
}
__syncthreads();
{ using WarpReduce = cub::WarpReduce<float>;
__shared__ typename WarpReduce::TempStorage temp_storage[HE_BLOCK_WARP_SIZE];
int row_tmp = info_tmp.row_st;
for (int col_iter = col_st; col_iter < col_ed; col_iter += blockDim.x) {
int nnz_tmp = col_iter + threadIdx.x;
int col_tmp = ColIdx[nnz_tmp];
int new_line_flag = 0; 
float res_tmp = 0;
if (nnz_tmp < col_ed) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
new_line_flag = (RowPtr[row_tmp] == nnz_tmp || lane_id == 0);
res_tmp = kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]);
}
else
res_tmp = -99999;
float reduce_tmp = WarpReduce(temp_storage[local_wid]).HeadSegmentedReduce(res_tmp, new_line_flag, cub::Max());
if (new_line_flag) atomicMax_float(&em[row_tmp * 1 + 0], reduce_tmp);
__syncwarp();
}
}
__syncthreads();
{ for (int row_tmp = row_tmp_ne_st; row_tmp < row_tmp_ne_ed; row_tmp += HE_BLOCK_WARP_SIZE) {
for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);
}
}
}
__syncthreads();
{ using WarpReduce = cub::WarpReduce<float>;
__shared__ typename WarpReduce::TempStorage temp_storage[HE_BLOCK_WARP_SIZE];
int row_tmp = info_tmp.row_st;
for (int col_iter = col_st; col_iter < col_ed; col_iter += blockDim.x) {
int nnz_tmp = col_iter + threadIdx.x;
int col_tmp = ColIdx[nnz_tmp];
int new_line_flag = 0; 
float res_tmp = 0;
if (nnz_tmp < col_ed) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
new_line_flag = (RowPtr[row_tmp] == nnz_tmp || lane_id == 0);
res_tmp = ho[nnz_tmp * 1 + 0];
}
float reduce_tmp = WarpReduce(temp_storage[local_wid]).HeadSegmentedReduce(res_tmp, new_line_flag, cub::Sum());
if (new_line_flag) atomicAdd(&es[row_tmp * 1 + 0], reduce_tmp);
__syncwarp();
}
}
__syncthreads();
}
else {
int row_tmp = info_tmp.row_st;
int nnzs = RowPtr[row_tmp + 1] - RowPtr[row_tmp];
int dynamic_blocks = (nnzs + DP_NNZ_PER_BLOCK - 1) / DP_NNZ_PER_BLOCK;
if (threadIdx.x == 0) k23_fop0_dp0<<<dynamic_blocks, DP_BLOCK_SIZE>>>(RowPtr, ColIdx, row_tmp, featlen, e, h);
{
float res_tmp = -99999;
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) {
atomicMax_float(&em[row_tmp * 1 + 0], res_tmp);
}
}
if (threadIdx.x == 0) k23_fop0_dp2<<<dynamic_blocks, DP_BLOCK_SIZE>>>(RowPtr, ColIdx, row_tmp, featlen, h, em, ho, es);
}
}
__global__ void k_23_fop_1(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* ho, float* f, float* es, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (global_wid >= *ng_n) return;
neighbor_group ng_tmp = ng_list[global_wid];
int row_tmp = ng_tmp.row_st;
int col_st = ng_tmp.col_st;
int col_ed = ng_tmp.col_st + NG_SIZE;
if (col_ed >= RowPtr[row_tmp + 1]) col_ed = RowPtr[row_tmp + 1];
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = col_st; nnz_tmp < col_ed; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
atomicAdd(&fo[row_tmp * featlen + k_tmp], res_tmp);
}
}
void gat_kernel_23(int64_t info_addr, torch::Tensor RowPtr, torch::Tensor ColIdx, int64_t featlen, torch::Tensor f, torch::Tensor fo, torch::Tensor we, torch::Tensor lr, torch::Tensor e, torch::Tensor em, torch::Tensor es, torch::Tensor h, torch::Tensor ho) {
kg_info* info_ = (kg_info*)info_addr;
if (info_ == nullptr) {
  throw std::runtime_error("info_ is null in gat_kernel_23");
}
if (info_->rp_info == nullptr || info_->rp_n == nullptr || info_->ep_info == nullptr || info_->ep_n == nullptr || info_->ng_info == nullptr || info_->ng_n == nullptr) {
  throw std::runtime_error("Some preprocessing pointers are null in gat_kernel_23");
}
int numnodes = RowPtr.size(0) - 1;
int numedges = ColIdx.size(0);
int num_n_block = (numnodes + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
int num_ne_block = (numnodes + NE_BLOCK_WARP_SIZE - 1) / NE_BLOCK_WARP_SIZE;
int num_e_block = info_->ep_n_host;
int num_he_block = info_->rp_n_host;
int num_ngd_block = (info_->ng_n_host + NGD_BLOCK_WARP_SIZE - 1) / NGD_BLOCK_WARP_SIZE;
k_23_fop_0<<<num_he_block, HE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, e.data_ptr<float>(), h.data_ptr<float>(), lr.data_ptr<float>(), em.data_ptr<float>(), ho.data_ptr<float>(), es.data_ptr<float>());
k_23_fop_1<<<num_ngd_block, NGD_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, ho.data_ptr<float>(), f.data_ptr<float>(), es.data_ptr<float>(), fo.data_ptr<float>());
}
__global__ void k_24_fop_0(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* e, float* h) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *ep_n) return;
row_panel info_tmp = ep_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (info_tmp.col_st == -1){ 
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
int row_tmp = info_tmp.row_st;
for (int nnz_tmp = col_st + threadIdx.x; nnz_tmp < col_ed; nnz_tmp += blockDim.x) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];
}
}
else {
int row_tmp = info_tmp.row_st;
for (int nnz_tmp = info_tmp.col_st + threadIdx.x; nnz_tmp < info_tmp.col_ed; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];}
}
}
__global__ void k_24_fop_1(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* h, float* lr, float* em) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *ep_n) return;
row_panel info_tmp = ep_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (info_tmp.col_st == -1){ 
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
using WarpReduce = cub::WarpReduce<float>;
__shared__ typename WarpReduce::TempStorage temp_storage[E_BLOCK_WARP_SIZE];
int row_tmp = info_tmp.row_st;
for (int col_iter = col_st; col_iter < col_ed; col_iter += blockDim.x) {
int nnz_tmp = col_iter + threadIdx.x;
int col_tmp = ColIdx[nnz_tmp];
int new_line_flag = 0; 
float res_tmp = 0;
if (nnz_tmp < col_ed) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
new_line_flag = (RowPtr[row_tmp] == nnz_tmp || lane_id == 0);
res_tmp = kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]);
}
else
res_tmp = -99999;
float reduce_tmp = WarpReduce(temp_storage[local_wid]).HeadSegmentedReduce(res_tmp, new_line_flag, cub::Max());
if (new_line_flag) atomicMax_float(&em[row_tmp * 1 + 0], reduce_tmp);
__syncwarp();
}
}
else {
int row_tmp = info_tmp.row_st;
float res_tmp = -99999;
for (int nnz_tmp = info_tmp.col_st + threadIdx.x; nnz_tmp < info_tmp.col_ed; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) {
atomicMax_float(&em[row_tmp * 1 + 0], res_tmp);
}
}
}
__global__ void k_24_fop_2(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* h, float* em, float* ho) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *ep_n) return;
row_panel info_tmp = ep_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (info_tmp.col_st == -1){ 
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
int row_tmp = info_tmp.row_st;
for (int nnz_tmp = col_st + threadIdx.x; nnz_tmp < col_ed; nnz_tmp += blockDim.x) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);
}
}
else {
int row_tmp = info_tmp.row_st;
for (int nnz_tmp = info_tmp.col_st + threadIdx.x; nnz_tmp < info_tmp.col_ed; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);}
}
}
__global__ void k_24_fop_3(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* ho, float* es) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *ep_n) return;
row_panel info_tmp = ep_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (info_tmp.col_st == -1){ 
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
using WarpReduce = cub::WarpReduce<float>;
__shared__ typename WarpReduce::TempStorage temp_storage[E_BLOCK_WARP_SIZE];
int row_tmp = info_tmp.row_st;
for (int col_iter = col_st; col_iter < col_ed; col_iter += blockDim.x) {
int nnz_tmp = col_iter + threadIdx.x;
int col_tmp = ColIdx[nnz_tmp];
int new_line_flag = 0; 
float res_tmp = 0;
if (nnz_tmp < col_ed) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
new_line_flag = (RowPtr[row_tmp] == nnz_tmp || lane_id == 0);
res_tmp = ho[nnz_tmp * 1 + 0];
}
float reduce_tmp = WarpReduce(temp_storage[local_wid]).HeadSegmentedReduce(res_tmp, new_line_flag, cub::Sum());
if (new_line_flag) atomicAdd(&es[row_tmp * 1 + 0], reduce_tmp);
__syncwarp();
}
}
else {
int row_tmp = info_tmp.row_st;
float res_tmp = 0;
for (int nnz_tmp = info_tmp.col_st + threadIdx.x; nnz_tmp < info_tmp.col_ed; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) {
atomicAdd(&es[row_tmp * 1 + 0], res_tmp);
}
}
}
__global__ void k_24_fop_4(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* ho, float* f, float* es, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
int row_tmp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
if (row_tmp >= numnodes) return;
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = RowPtr[row_tmp]; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
fo[row_tmp * featlen + k_tmp] = res_tmp;
}
}
void gat_kernel_24(int64_t info_addr, torch::Tensor RowPtr, torch::Tensor ColIdx, int64_t featlen, torch::Tensor f, torch::Tensor fo, torch::Tensor we, torch::Tensor lr, torch::Tensor e, torch::Tensor em, torch::Tensor es, torch::Tensor h, torch::Tensor ho) {
kg_info* info_ = (kg_info*)info_addr;
if (info_ == nullptr) {
  throw std::runtime_error("info_ is null in gat_kernel_24");
}
if (info_->rp_info == nullptr || info_->rp_n == nullptr || info_->ep_info == nullptr || info_->ep_n == nullptr || info_->ng_info == nullptr || info_->ng_n == nullptr) {
  throw std::runtime_error("Some preprocessing pointers are null in gat_kernel_24");
}
int numnodes = RowPtr.size(0) - 1;
int numedges = ColIdx.size(0);
int num_n_block = (numnodes + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
int num_ne_block = (numnodes + NE_BLOCK_WARP_SIZE - 1) / NE_BLOCK_WARP_SIZE;
int num_e_block = info_->ep_n_host;
int num_he_block = info_->rp_n_host;
int num_ngd_block = (info_->ng_n_host + NGD_BLOCK_WARP_SIZE - 1) / NGD_BLOCK_WARP_SIZE;
k_24_fop_0<<<num_e_block, E_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, e.data_ptr<float>(), h.data_ptr<float>());
k_24_fop_1<<<num_e_block, E_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, h.data_ptr<float>(), lr.data_ptr<float>(), em.data_ptr<float>());
k_24_fop_2<<<num_e_block, E_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, h.data_ptr<float>(), em.data_ptr<float>(), ho.data_ptr<float>());
k_24_fop_3<<<num_e_block, E_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, ho.data_ptr<float>(), es.data_ptr<float>());
k_24_fop_4<<<num_ne_block, NE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, ho.data_ptr<float>(), f.data_ptr<float>(), es.data_ptr<float>(), fo.data_ptr<float>());
}
__global__ void k25_fop0_dp0(int *RowPtr, int *ColIdx, int row_tmp, int featlen, float* e, float* h) {
int lane_id = threadIdx.x % WARP_SIZE;
int idx_st = RowPtr[row_tmp] + blockIdx.x * DP_NNZ_PER_BLOCK + threadIdx.x / WARP_SIZE * DP_NNZ_PER_WARP;
int idx_ed = idx_st + DP_NNZ_PER_WARP;
if (idx_ed > RowPtr[row_tmp + 1]) idx_ed = RowPtr[row_tmp + 1];
for (int nnz_tmp = idx_st + lane_id; nnz_tmp < idx_ed; nnz_tmp += WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];}
}
__global__ void k25_fop0_dp2(int *RowPtr, int *ColIdx, int row_tmp, int featlen, float* h, float* em, float* ho, float* es) {
int lane_id = threadIdx.x % WARP_SIZE;
int idx_st = RowPtr[row_tmp] + blockIdx.x * DP_NNZ_PER_BLOCK + threadIdx.x / WARP_SIZE * DP_NNZ_PER_WARP;
int idx_ed = idx_st + DP_NNZ_PER_WARP;
if (idx_ed > RowPtr[row_tmp + 1]) idx_ed = RowPtr[row_tmp + 1];
for (int nnz_tmp = idx_st + lane_id; nnz_tmp < idx_ed; nnz_tmp += WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);}
float res_tmp = 0;
for (int nnz_tmp = idx_st + lane_id; nnz_tmp < idx_ed; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) {
atomicAdd(&es[row_tmp * 1 + 0], res_tmp);
}
}
__global__ void k_25_fop_0(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* e, float* h, float* lr, float* em, float* ho, float* es) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *info_n) return;
row_panel info_tmp = info_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
if (info_tmp.col_st == -1) {
{ { int row_tmp = info_tmp.row_st;
for (int nnz_tmp = col_st + threadIdx.x; nnz_tmp < col_ed; nnz_tmp += HE_BLOCK_SIZE) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];
}
}
}
__syncthreads();
{ using WarpReduce = cub::WarpReduce<float>;
__shared__ typename WarpReduce::TempStorage temp_storage[HE_BLOCK_WARP_SIZE];
int row_tmp = info_tmp.row_st;
for (int col_iter = col_st; col_iter < col_ed; col_iter += blockDim.x) {
int nnz_tmp = col_iter + threadIdx.x;
int col_tmp = ColIdx[nnz_tmp];
int new_line_flag = 0; 
float res_tmp = 0;
if (nnz_tmp < col_ed) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
new_line_flag = (RowPtr[row_tmp] == nnz_tmp || lane_id == 0);
res_tmp = kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]);
}
else
res_tmp = -99999;
float reduce_tmp = WarpReduce(temp_storage[local_wid]).HeadSegmentedReduce(res_tmp, new_line_flag, cub::Max());
if (new_line_flag) atomicMax_float(&em[row_tmp * 1 + 0], reduce_tmp);
__syncwarp();
}
}
__syncthreads();
{ { int row_tmp = info_tmp.row_st;
for (int nnz_tmp = col_st + threadIdx.x; nnz_tmp < col_ed; nnz_tmp += HE_BLOCK_SIZE) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);
}
}
}
__syncthreads();
{ using WarpReduce = cub::WarpReduce<float>;
__shared__ typename WarpReduce::TempStorage temp_storage[HE_BLOCK_WARP_SIZE];
int row_tmp = info_tmp.row_st;
for (int col_iter = col_st; col_iter < col_ed; col_iter += blockDim.x) {
int nnz_tmp = col_iter + threadIdx.x;
int col_tmp = ColIdx[nnz_tmp];
int new_line_flag = 0; 
float res_tmp = 0;
if (nnz_tmp < col_ed) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
new_line_flag = (RowPtr[row_tmp] == nnz_tmp || lane_id == 0);
res_tmp = ho[nnz_tmp * 1 + 0];
}
float reduce_tmp = WarpReduce(temp_storage[local_wid]).HeadSegmentedReduce(res_tmp, new_line_flag, cub::Sum());
if (new_line_flag) atomicAdd(&es[row_tmp * 1 + 0], reduce_tmp);
__syncwarp();
}
}
__syncthreads();
}
else {
int row_tmp = info_tmp.row_st;
int nnzs = RowPtr[row_tmp + 1] - RowPtr[row_tmp];
int dynamic_blocks = (nnzs + DP_NNZ_PER_BLOCK - 1) / DP_NNZ_PER_BLOCK;
if (threadIdx.x == 0) k25_fop0_dp0<<<dynamic_blocks, DP_BLOCK_SIZE>>>(RowPtr, ColIdx, row_tmp, featlen, e, h);
{
float res_tmp = -99999;
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) {
atomicMax_float(&em[row_tmp * 1 + 0], res_tmp);
}
}
if (threadIdx.x == 0) k25_fop0_dp2<<<dynamic_blocks, DP_BLOCK_SIZE>>>(RowPtr, ColIdx, row_tmp, featlen, h, em, ho, es);
}
}
__global__ void k_25_fop_1(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* ho, float* f, float* es, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
int row_tmp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
if (row_tmp >= numnodes) return;
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = RowPtr[row_tmp]; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
fo[row_tmp * featlen + k_tmp] = res_tmp;
}
}
void gat_kernel_25(int64_t info_addr, torch::Tensor RowPtr, torch::Tensor ColIdx, int64_t featlen, torch::Tensor f, torch::Tensor fo, torch::Tensor we, torch::Tensor lr, torch::Tensor e, torch::Tensor em, torch::Tensor es, torch::Tensor h, torch::Tensor ho) {
kg_info* info_ = (kg_info*)info_addr;
if (info_ == nullptr) {
  throw std::runtime_error("info_ is null in gat_kernel_25");
}
if (info_->rp_info == nullptr || info_->rp_n == nullptr || info_->ep_info == nullptr || info_->ep_n == nullptr || info_->ng_info == nullptr || info_->ng_n == nullptr) {
  throw std::runtime_error("Some preprocessing pointers are null in gat_kernel_25");
}
int numnodes = RowPtr.size(0) - 1;
int numedges = ColIdx.size(0);
int num_n_block = (numnodes + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
int num_ne_block = (numnodes + NE_BLOCK_WARP_SIZE - 1) / NE_BLOCK_WARP_SIZE;
int num_e_block = info_->ep_n_host;
int num_he_block = info_->rp_n_host;
int num_ngd_block = (info_->ng_n_host + NGD_BLOCK_WARP_SIZE - 1) / NGD_BLOCK_WARP_SIZE;
k_25_fop_0<<<num_he_block, HE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, e.data_ptr<float>(), h.data_ptr<float>(), lr.data_ptr<float>(), em.data_ptr<float>(), ho.data_ptr<float>(), es.data_ptr<float>());
k_25_fop_1<<<num_ne_block, NE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, ho.data_ptr<float>(), f.data_ptr<float>(), es.data_ptr<float>(), fo.data_ptr<float>());
}
__global__ void k26_fop0_dp4(int *RowPtr, int *ColIdx, int row_tmp, int featlen, float* ho, float* f, float* es, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
int idx_st = RowPtr[row_tmp] + blockIdx.x * DP_NNZ_PER_BLOCK + threadIdx.x / WARP_SIZE * DP_NNZ_PER_WARP;
int idx_ed = idx_st + DP_NNZ_PER_WARP;
if (idx_ed > RowPtr[row_tmp + 1]) idx_ed = RowPtr[row_tmp + 1];
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = idx_st; nnz_tmp < idx_ed; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];
}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
atomicAdd(&fo[row_tmp * featlen + k_tmp], res_tmp);
}
}
__global__ void k_26_fop_0(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* e, float* h, float* lr, float* em, float* ho, float* es, float* f, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *info_n) return;
row_panel info_tmp = info_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
int row_tmp_ne_st = info_tmp.row_st + local_wid;
int row_tmp_ne_ed = info_tmp.row_ed;
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
if (info_tmp.col_st == -1) {
{ { int row_tmp = info_tmp.row_st;
for (int nnz_tmp = col_st + threadIdx.x; nnz_tmp < col_ed; nnz_tmp += HE_BLOCK_SIZE) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];
}
}
}
__syncthreads();
{ using WarpReduce = cub::WarpReduce<float>;
__shared__ typename WarpReduce::TempStorage temp_storage[HE_BLOCK_WARP_SIZE];
int row_tmp = info_tmp.row_st;
for (int col_iter = col_st; col_iter < col_ed; col_iter += blockDim.x) {
int nnz_tmp = col_iter + threadIdx.x;
int col_tmp = ColIdx[nnz_tmp];
int new_line_flag = 0; 
float res_tmp = 0;
if (nnz_tmp < col_ed) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
new_line_flag = (RowPtr[row_tmp] == nnz_tmp || lane_id == 0);
res_tmp = kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]);
}
else
res_tmp = -99999;
float reduce_tmp = WarpReduce(temp_storage[local_wid]).HeadSegmentedReduce(res_tmp, new_line_flag, cub::Max());
if (new_line_flag) atomicMax_float(&em[row_tmp * 1 + 0], reduce_tmp);
__syncwarp();
}
}
__syncthreads();
{ { int row_tmp = info_tmp.row_st;
for (int nnz_tmp = col_st + threadIdx.x; nnz_tmp < col_ed; nnz_tmp += HE_BLOCK_SIZE) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);
}
}
}
__syncthreads();
{ using WarpReduce = cub::WarpReduce<float>;
__shared__ typename WarpReduce::TempStorage temp_storage[HE_BLOCK_WARP_SIZE];
int row_tmp = info_tmp.row_st;
for (int col_iter = col_st; col_iter < col_ed; col_iter += blockDim.x) {
int nnz_tmp = col_iter + threadIdx.x;
int col_tmp = ColIdx[nnz_tmp];
int new_line_flag = 0; 
float res_tmp = 0;
if (nnz_tmp < col_ed) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
new_line_flag = (RowPtr[row_tmp] == nnz_tmp || lane_id == 0);
res_tmp = ho[nnz_tmp * 1 + 0];
}
float reduce_tmp = WarpReduce(temp_storage[local_wid]).HeadSegmentedReduce(res_tmp, new_line_flag, cub::Sum());
if (new_line_flag) atomicAdd(&es[row_tmp * 1 + 0], reduce_tmp);
__syncwarp();
}
}
__syncthreads();
{ for (int row_tmp = row_tmp_ne_st; row_tmp < row_tmp_ne_ed; row_tmp += HE_BLOCK_WARP_SIZE) {
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = RowPtr[row_tmp]; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];
}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
fo[row_tmp * featlen + k_tmp] = res_tmp;
}
}
}
__syncthreads();
}
else {
int row_tmp = info_tmp.row_st;
int nnzs = RowPtr[row_tmp + 1] - RowPtr[row_tmp];
int dynamic_blocks = (nnzs + DP_NNZ_PER_BLOCK - 1) / DP_NNZ_PER_BLOCK;
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];}__syncthreads();
{
float res_tmp = -99999;
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) {
atomicMax_float(&em[row_tmp * 1 + 0], res_tmp);
}
}__syncthreads();
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);}__syncthreads();
{
float res_tmp = 0;
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) {
atomicAdd(&es[row_tmp * 1 + 0], res_tmp);
}
}__syncthreads();
if (threadIdx.x == 0) k26_fop0_dp4<<<dynamic_blocks, DP_BLOCK_SIZE>>>(RowPtr, ColIdx, row_tmp, featlen, ho, f, es, fo);
}
}
void gat_kernel_26(int64_t info_addr, torch::Tensor RowPtr, torch::Tensor ColIdx, int64_t featlen, torch::Tensor f, torch::Tensor fo, torch::Tensor we, torch::Tensor lr, torch::Tensor e, torch::Tensor em, torch::Tensor es, torch::Tensor h, torch::Tensor ho) {
kg_info* info_ = (kg_info*)info_addr;
if (info_ == nullptr) {
  throw std::runtime_error("info_ is null in gat_kernel_26");
}
if (info_->rp_info == nullptr || info_->rp_n == nullptr || info_->ep_info == nullptr || info_->ep_n == nullptr || info_->ng_info == nullptr || info_->ng_n == nullptr) {
  throw std::runtime_error("Some preprocessing pointers are null in gat_kernel_26");
}
int numnodes = RowPtr.size(0) - 1;
int numedges = ColIdx.size(0);
int num_n_block = (numnodes + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
int num_ne_block = (numnodes + NE_BLOCK_WARP_SIZE - 1) / NE_BLOCK_WARP_SIZE;
int num_e_block = info_->ep_n_host;
int num_he_block = info_->rp_n_host;
int num_ngd_block = (info_->ng_n_host + NGD_BLOCK_WARP_SIZE - 1) / NGD_BLOCK_WARP_SIZE;
k_26_fop_0<<<num_he_block, HE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, e.data_ptr<float>(), h.data_ptr<float>(), lr.data_ptr<float>(), em.data_ptr<float>(), ho.data_ptr<float>(), es.data_ptr<float>(), f.data_ptr<float>(), fo.data_ptr<float>());
}
__global__ void k_27_fop_0(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* e, float* h) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *ep_n) return;
row_panel info_tmp = ep_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (info_tmp.col_st == -1){ 
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
int row_tmp = info_tmp.row_st;
for (int nnz_tmp = col_st + threadIdx.x; nnz_tmp < col_ed; nnz_tmp += blockDim.x) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];
}
}
else {
int row_tmp = info_tmp.row_st;
for (int nnz_tmp = info_tmp.col_st + threadIdx.x; nnz_tmp < info_tmp.col_ed; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];}
}
}
__global__ void k_27_fop_1(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* h, float* lr, float* em) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *ep_n) return;
row_panel info_tmp = ep_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (info_tmp.col_st == -1){ 
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
using WarpReduce = cub::WarpReduce<float>;
__shared__ typename WarpReduce::TempStorage temp_storage[E_BLOCK_WARP_SIZE];
int row_tmp = info_tmp.row_st;
for (int col_iter = col_st; col_iter < col_ed; col_iter += blockDim.x) {
int nnz_tmp = col_iter + threadIdx.x;
int col_tmp = ColIdx[nnz_tmp];
int new_line_flag = 0; 
float res_tmp = 0;
if (nnz_tmp < col_ed) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
new_line_flag = (RowPtr[row_tmp] == nnz_tmp || lane_id == 0);
res_tmp = kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]);
}
else
res_tmp = -99999;
float reduce_tmp = WarpReduce(temp_storage[local_wid]).HeadSegmentedReduce(res_tmp, new_line_flag, cub::Max());
if (new_line_flag) atomicMax_float(&em[row_tmp * 1 + 0], reduce_tmp);
__syncwarp();
}
}
else {
int row_tmp = info_tmp.row_st;
float res_tmp = -99999;
for (int nnz_tmp = info_tmp.col_st + threadIdx.x; nnz_tmp < info_tmp.col_ed; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) {
atomicMax_float(&em[row_tmp * 1 + 0], res_tmp);
}
}
}
__global__ void k_27_fop_2(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* h, float* em, float* ho) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *ep_n) return;
row_panel info_tmp = ep_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (info_tmp.col_st == -1){ 
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
int row_tmp = info_tmp.row_st;
for (int nnz_tmp = col_st + threadIdx.x; nnz_tmp < col_ed; nnz_tmp += blockDim.x) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);
}
}
else {
int row_tmp = info_tmp.row_st;
for (int nnz_tmp = info_tmp.col_st + threadIdx.x; nnz_tmp < info_tmp.col_ed; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);}
}
}
__global__ void k_27_fop_3(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* ho, float* es) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *ep_n) return;
row_panel info_tmp = ep_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (info_tmp.col_st == -1){ 
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
using WarpReduce = cub::WarpReduce<float>;
__shared__ typename WarpReduce::TempStorage temp_storage[E_BLOCK_WARP_SIZE];
int row_tmp = info_tmp.row_st;
for (int col_iter = col_st; col_iter < col_ed; col_iter += blockDim.x) {
int nnz_tmp = col_iter + threadIdx.x;
int col_tmp = ColIdx[nnz_tmp];
int new_line_flag = 0; 
float res_tmp = 0;
if (nnz_tmp < col_ed) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
new_line_flag = (RowPtr[row_tmp] == nnz_tmp || lane_id == 0);
res_tmp = ho[nnz_tmp * 1 + 0];
}
float reduce_tmp = WarpReduce(temp_storage[local_wid]).HeadSegmentedReduce(res_tmp, new_line_flag, cub::Sum());
if (new_line_flag) atomicAdd(&es[row_tmp * 1 + 0], reduce_tmp);
__syncwarp();
}
}
else {
int row_tmp = info_tmp.row_st;
float res_tmp = 0;
for (int nnz_tmp = info_tmp.col_st + threadIdx.x; nnz_tmp < info_tmp.col_ed; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) {
atomicAdd(&es[row_tmp * 1 + 0], res_tmp);
}
}
}
__global__ void k_27_fop_4(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* ho, float* f, float* es, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *ep_n) return;
row_panel info_tmp = ep_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (info_tmp.col_st == -1){ 
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
int nnz_per_warp = (nnzs + E_BLOCK_WARP_SIZE - 1) / E_BLOCK_WARP_SIZE;
int warp_col_st = col_st + nnz_per_warp * local_wid;
int warp_col_ed = (warp_col_st + nnz_per_warp < col_ed)? warp_col_st + nnz_per_warp: col_ed;
int warp_row_st = info_tmp.row_st, warp_row_ed;
while (warp_row_st < info_tmp.row_ed - 1 && warp_col_st > RowPtr[warp_row_st + 1]) {
warp_row_st++;
}
warp_row_ed = warp_row_st;
while (warp_row_ed < info_tmp.row_ed && warp_col_ed > RowPtr[warp_row_ed]) {
warp_row_ed++;
}
if (warp_col_ed <= col_ed) {
for (int row_tmp = warp_row_st; row_tmp < warp_row_ed; row_tmp++) {
int current_col_st = RowPtr[row_tmp];
int current_col_ed = RowPtr[row_tmp + 1];
if (row_tmp == warp_row_st) current_col_st = warp_col_st;
if (row_tmp == warp_row_ed - 1) current_col_ed = warp_col_ed;
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = current_col_st; nnz_tmp < current_col_ed; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
atomicAdd(&fo[row_tmp * featlen + k_tmp], res_tmp);
}}}
}
else {
int row_tmp = info_tmp.row_st;
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = info_tmp.col_st + local_wid; nnz_tmp < info_tmp.col_ed; nnz_tmp+=HE_BLOCK_WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];
}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
atomicAdd(&fo[row_tmp * featlen + k_tmp], res_tmp);
}
}
}
void gat_kernel_27(int64_t info_addr, torch::Tensor RowPtr, torch::Tensor ColIdx, int64_t featlen, torch::Tensor f, torch::Tensor fo, torch::Tensor we, torch::Tensor lr, torch::Tensor e, torch::Tensor em, torch::Tensor es, torch::Tensor h, torch::Tensor ho) {
kg_info* info_ = (kg_info*)info_addr;
if (info_ == nullptr) {
  throw std::runtime_error("info_ is null in gat_kernel_27");
}
if (info_->rp_info == nullptr || info_->rp_n == nullptr || info_->ep_info == nullptr || info_->ep_n == nullptr || info_->ng_info == nullptr || info_->ng_n == nullptr) {
  throw std::runtime_error("Some preprocessing pointers are null in gat_kernel_27");
}
int numnodes = RowPtr.size(0) - 1;
int numedges = ColIdx.size(0);
int num_n_block = (numnodes + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
int num_ne_block = (numnodes + NE_BLOCK_WARP_SIZE - 1) / NE_BLOCK_WARP_SIZE;
int num_e_block = info_->ep_n_host;
int num_he_block = info_->rp_n_host;
int num_ngd_block = (info_->ng_n_host + NGD_BLOCK_WARP_SIZE - 1) / NGD_BLOCK_WARP_SIZE;
k_27_fop_0<<<num_e_block, E_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, e.data_ptr<float>(), h.data_ptr<float>());
k_27_fop_1<<<num_e_block, E_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, h.data_ptr<float>(), lr.data_ptr<float>(), em.data_ptr<float>());
k_27_fop_2<<<num_e_block, E_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, h.data_ptr<float>(), em.data_ptr<float>(), ho.data_ptr<float>());
k_27_fop_3<<<num_e_block, E_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, ho.data_ptr<float>(), es.data_ptr<float>());
k_27_fop_4<<<num_e_block, E_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, ho.data_ptr<float>(), f.data_ptr<float>(), es.data_ptr<float>(), fo.data_ptr<float>());
}
__global__ void k28_fop0_dp0(int *RowPtr, int *ColIdx, int row_tmp, int featlen, float* e, float* h) {
int lane_id = threadIdx.x % WARP_SIZE;
int idx_st = RowPtr[row_tmp] + blockIdx.x * DP_NNZ_PER_BLOCK + threadIdx.x / WARP_SIZE * DP_NNZ_PER_WARP;
int idx_ed = idx_st + DP_NNZ_PER_WARP;
if (idx_ed > RowPtr[row_tmp + 1]) idx_ed = RowPtr[row_tmp + 1];
for (int nnz_tmp = idx_st + lane_id; nnz_tmp < idx_ed; nnz_tmp += WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];}
}
__global__ void k28_fop0_dp2(int *RowPtr, int *ColIdx, int row_tmp, int featlen, float* h, float* em, float* ho, float* es) {
int lane_id = threadIdx.x % WARP_SIZE;
int idx_st = RowPtr[row_tmp] + blockIdx.x * DP_NNZ_PER_BLOCK + threadIdx.x / WARP_SIZE * DP_NNZ_PER_WARP;
int idx_ed = idx_st + DP_NNZ_PER_WARP;
if (idx_ed > RowPtr[row_tmp + 1]) idx_ed = RowPtr[row_tmp + 1];
for (int nnz_tmp = idx_st + lane_id; nnz_tmp < idx_ed; nnz_tmp += WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);}
float res_tmp = 0;
for (int nnz_tmp = idx_st + lane_id; nnz_tmp < idx_ed; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) {
atomicAdd(&es[row_tmp * 1 + 0], res_tmp);
}
}
__global__ void k_28_fop_0(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* e, float* h, float* lr, float* em, float* ho, float* es) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *info_n) return;
row_panel info_tmp = info_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
if (info_tmp.col_st == -1) {
{ { int row_tmp = info_tmp.row_st;
for (int nnz_tmp = col_st + threadIdx.x; nnz_tmp < col_ed; nnz_tmp += HE_BLOCK_SIZE) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];
}
}
}
__syncthreads();
{ using WarpReduce = cub::WarpReduce<float>;
__shared__ typename WarpReduce::TempStorage temp_storage[HE_BLOCK_WARP_SIZE];
int row_tmp = info_tmp.row_st;
for (int col_iter = col_st; col_iter < col_ed; col_iter += blockDim.x) {
int nnz_tmp = col_iter + threadIdx.x;
int col_tmp = ColIdx[nnz_tmp];
int new_line_flag = 0; 
float res_tmp = 0;
if (nnz_tmp < col_ed) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
new_line_flag = (RowPtr[row_tmp] == nnz_tmp || lane_id == 0);
res_tmp = kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]);
}
else
res_tmp = -99999;
float reduce_tmp = WarpReduce(temp_storage[local_wid]).HeadSegmentedReduce(res_tmp, new_line_flag, cub::Max());
if (new_line_flag) atomicMax_float(&em[row_tmp * 1 + 0], reduce_tmp);
__syncwarp();
}
}
__syncthreads();
{ { int row_tmp = info_tmp.row_st;
for (int nnz_tmp = col_st + threadIdx.x; nnz_tmp < col_ed; nnz_tmp += HE_BLOCK_SIZE) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);
}
}
}
__syncthreads();
{ using WarpReduce = cub::WarpReduce<float>;
__shared__ typename WarpReduce::TempStorage temp_storage[HE_BLOCK_WARP_SIZE];
int row_tmp = info_tmp.row_st;
for (int col_iter = col_st; col_iter < col_ed; col_iter += blockDim.x) {
int nnz_tmp = col_iter + threadIdx.x;
int col_tmp = ColIdx[nnz_tmp];
int new_line_flag = 0; 
float res_tmp = 0;
if (nnz_tmp < col_ed) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
new_line_flag = (RowPtr[row_tmp] == nnz_tmp || lane_id == 0);
res_tmp = ho[nnz_tmp * 1 + 0];
}
float reduce_tmp = WarpReduce(temp_storage[local_wid]).HeadSegmentedReduce(res_tmp, new_line_flag, cub::Sum());
if (new_line_flag) atomicAdd(&es[row_tmp * 1 + 0], reduce_tmp);
__syncwarp();
}
}
__syncthreads();
}
else {
int row_tmp = info_tmp.row_st;
int nnzs = RowPtr[row_tmp + 1] - RowPtr[row_tmp];
int dynamic_blocks = (nnzs + DP_NNZ_PER_BLOCK - 1) / DP_NNZ_PER_BLOCK;
if (threadIdx.x == 0) k28_fop0_dp0<<<dynamic_blocks, DP_BLOCK_SIZE>>>(RowPtr, ColIdx, row_tmp, featlen, e, h);
{
float res_tmp = -99999;
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) {
atomicMax_float(&em[row_tmp * 1 + 0], res_tmp);
}
}
if (threadIdx.x == 0) k28_fop0_dp2<<<dynamic_blocks, DP_BLOCK_SIZE>>>(RowPtr, ColIdx, row_tmp, featlen, h, em, ho, es);
}
}
__global__ void k_28_fop_1(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* ho, float* f, float* es, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *ep_n) return;
row_panel info_tmp = ep_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (info_tmp.col_st == -1){ 
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
int nnz_per_warp = (nnzs + E_BLOCK_WARP_SIZE - 1) / E_BLOCK_WARP_SIZE;
int warp_col_st = col_st + nnz_per_warp * local_wid;
int warp_col_ed = (warp_col_st + nnz_per_warp < col_ed)? warp_col_st + nnz_per_warp: col_ed;
int warp_row_st = info_tmp.row_st, warp_row_ed;
while (warp_row_st < info_tmp.row_ed - 1 && warp_col_st > RowPtr[warp_row_st + 1]) {
warp_row_st++;
}
warp_row_ed = warp_row_st;
while (warp_row_ed < info_tmp.row_ed && warp_col_ed > RowPtr[warp_row_ed]) {
warp_row_ed++;
}
if (warp_col_ed <= col_ed) {
for (int row_tmp = warp_row_st; row_tmp < warp_row_ed; row_tmp++) {
int current_col_st = RowPtr[row_tmp];
int current_col_ed = RowPtr[row_tmp + 1];
if (row_tmp == warp_row_st) current_col_st = warp_col_st;
if (row_tmp == warp_row_ed - 1) current_col_ed = warp_col_ed;
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = current_col_st; nnz_tmp < current_col_ed; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
atomicAdd(&fo[row_tmp * featlen + k_tmp], res_tmp);
}}}
}
else {
int row_tmp = info_tmp.row_st;
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = info_tmp.col_st + local_wid; nnz_tmp < info_tmp.col_ed; nnz_tmp+=HE_BLOCK_WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];
}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
atomicAdd(&fo[row_tmp * featlen + k_tmp], res_tmp);
}
}
}
void gat_kernel_28(int64_t info_addr, torch::Tensor RowPtr, torch::Tensor ColIdx, int64_t featlen, torch::Tensor f, torch::Tensor fo, torch::Tensor we, torch::Tensor lr, torch::Tensor e, torch::Tensor em, torch::Tensor es, torch::Tensor h, torch::Tensor ho) {
kg_info* info_ = (kg_info*)info_addr;
if (info_ == nullptr) {
  throw std::runtime_error("info_ is null in gat_kernel_28");
}
if (info_->rp_info == nullptr || info_->rp_n == nullptr || info_->ep_info == nullptr || info_->ep_n == nullptr || info_->ng_info == nullptr || info_->ng_n == nullptr) {
  throw std::runtime_error("Some preprocessing pointers are null in gat_kernel_28");
}
int numnodes = RowPtr.size(0) - 1;
int numedges = ColIdx.size(0);
int num_n_block = (numnodes + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
int num_ne_block = (numnodes + NE_BLOCK_WARP_SIZE - 1) / NE_BLOCK_WARP_SIZE;
int num_e_block = info_->ep_n_host;
int num_he_block = info_->rp_n_host;
int num_ngd_block = (info_->ng_n_host + NGD_BLOCK_WARP_SIZE - 1) / NGD_BLOCK_WARP_SIZE;
k_28_fop_0<<<num_he_block, HE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, e.data_ptr<float>(), h.data_ptr<float>(), lr.data_ptr<float>(), em.data_ptr<float>(), ho.data_ptr<float>(), es.data_ptr<float>());
k_28_fop_1<<<num_e_block, E_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, ho.data_ptr<float>(), f.data_ptr<float>(), es.data_ptr<float>(), fo.data_ptr<float>());
}
__global__ void k29_fop0_dp4(int *RowPtr, int *ColIdx, int row_tmp, int featlen, float* ho, float* f, float* es, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
int idx_st = RowPtr[row_tmp] + blockIdx.x * DP_NNZ_PER_BLOCK + threadIdx.x / WARP_SIZE * DP_NNZ_PER_WARP;
int idx_ed = idx_st + DP_NNZ_PER_WARP;
if (idx_ed > RowPtr[row_tmp + 1]) idx_ed = RowPtr[row_tmp + 1];
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = idx_st; nnz_tmp < idx_ed; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];
}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
atomicAdd(&fo[row_tmp * featlen + k_tmp], res_tmp);
}
}
__global__ void k_29_fop_0(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* e, float* h, float* lr, float* em, float* ho, float* es, float* f, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *info_n) return;
row_panel info_tmp = info_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
int nnz_per_warp = (nnzs + HE_BLOCK_WARP_SIZE - 1) / HE_BLOCK_WARP_SIZE;
int warp_col_st = col_st + nnz_per_warp * local_wid;
int warp_col_ed = (warp_col_st + nnz_per_warp < col_ed)? warp_col_st + nnz_per_warp: col_ed;
int warp_row_st = info_tmp.row_st, warp_row_ed;
while (warp_row_st < info_tmp.row_ed - 1 && warp_col_st > RowPtr[warp_row_st + 1]) {
warp_row_st++;
}
warp_row_ed = warp_row_st;
while (warp_row_ed < info_tmp.row_ed && warp_col_ed > RowPtr[warp_row_ed]) {
warp_row_ed++;
}
if (info_tmp.col_st == -1) {
{ { int row_tmp = info_tmp.row_st;
for (int nnz_tmp = col_st + threadIdx.x; nnz_tmp < col_ed; nnz_tmp += HE_BLOCK_SIZE) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];
}
}
}
__syncthreads();
{ using WarpReduce = cub::WarpReduce<float>;
__shared__ typename WarpReduce::TempStorage temp_storage[HE_BLOCK_WARP_SIZE];
int row_tmp = info_tmp.row_st;
for (int col_iter = col_st; col_iter < col_ed; col_iter += blockDim.x) {
int nnz_tmp = col_iter + threadIdx.x;
int col_tmp = ColIdx[nnz_tmp];
int new_line_flag = 0; 
float res_tmp = 0;
if (nnz_tmp < col_ed) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
new_line_flag = (RowPtr[row_tmp] == nnz_tmp || lane_id == 0);
res_tmp = kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]);
}
else
res_tmp = -99999;
float reduce_tmp = WarpReduce(temp_storage[local_wid]).HeadSegmentedReduce(res_tmp, new_line_flag, cub::Max());
if (new_line_flag) atomicMax_float(&em[row_tmp * 1 + 0], reduce_tmp);
__syncwarp();
}
}
__syncthreads();
{ { int row_tmp = info_tmp.row_st;
for (int nnz_tmp = col_st + threadIdx.x; nnz_tmp < col_ed; nnz_tmp += HE_BLOCK_SIZE) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);
}
}
}
__syncthreads();
{ using WarpReduce = cub::WarpReduce<float>;
__shared__ typename WarpReduce::TempStorage temp_storage[HE_BLOCK_WARP_SIZE];
int row_tmp = info_tmp.row_st;
for (int col_iter = col_st; col_iter < col_ed; col_iter += blockDim.x) {
int nnz_tmp = col_iter + threadIdx.x;
int col_tmp = ColIdx[nnz_tmp];
int new_line_flag = 0; 
float res_tmp = 0;
if (nnz_tmp < col_ed) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
new_line_flag = (RowPtr[row_tmp] == nnz_tmp || lane_id == 0);
res_tmp = ho[nnz_tmp * 1 + 0];
}
float reduce_tmp = WarpReduce(temp_storage[local_wid]).HeadSegmentedReduce(res_tmp, new_line_flag, cub::Sum());
if (new_line_flag) atomicAdd(&es[row_tmp * 1 + 0], reduce_tmp);
__syncwarp();
}
}
__syncthreads();
{ if (warp_col_ed <= col_ed) {
for (int row_tmp = warp_row_st; row_tmp < warp_row_ed; row_tmp++) {
int current_col_st = RowPtr[row_tmp];
int current_col_ed = RowPtr[row_tmp + 1];
if (row_tmp == warp_row_st) current_col_st = warp_col_st;
if (row_tmp == warp_row_ed - 1) current_col_ed = warp_col_ed;
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = current_col_st; nnz_tmp < current_col_ed; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];
}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
atomicAdd(&fo[row_tmp * featlen + k_tmp], res_tmp);
}}}
}
__syncthreads();
}
else {
int row_tmp = info_tmp.row_st;
int nnzs = RowPtr[row_tmp + 1] - RowPtr[row_tmp];
int dynamic_blocks = (nnzs + DP_NNZ_PER_BLOCK - 1) / DP_NNZ_PER_BLOCK;
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];}__syncthreads();
{
float res_tmp = -99999;
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) {
atomicMax_float(&em[row_tmp * 1 + 0], res_tmp);
}
}__syncthreads();
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);}__syncthreads();
{
float res_tmp = 0;
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) {
atomicAdd(&es[row_tmp * 1 + 0], res_tmp);
}
}__syncthreads();
if (threadIdx.x == 0) k29_fop0_dp4<<<dynamic_blocks, DP_BLOCK_SIZE>>>(RowPtr, ColIdx, row_tmp, featlen, ho, f, es, fo);
}
}
void gat_kernel_29(int64_t info_addr, torch::Tensor RowPtr, torch::Tensor ColIdx, int64_t featlen, torch::Tensor f, torch::Tensor fo, torch::Tensor we, torch::Tensor lr, torch::Tensor e, torch::Tensor em, torch::Tensor es, torch::Tensor h, torch::Tensor ho) {
kg_info* info_ = (kg_info*)info_addr;
if (info_ == nullptr) {
  throw std::runtime_error("info_ is null in gat_kernel_29");
}
if (info_->rp_info == nullptr || info_->rp_n == nullptr || info_->ep_info == nullptr || info_->ep_n == nullptr || info_->ng_info == nullptr || info_->ng_n == nullptr) {
  throw std::runtime_error("Some preprocessing pointers are null in gat_kernel_29");
}
int numnodes = RowPtr.size(0) - 1;
int numedges = ColIdx.size(0);
int num_n_block = (numnodes + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
int num_ne_block = (numnodes + NE_BLOCK_WARP_SIZE - 1) / NE_BLOCK_WARP_SIZE;
int num_e_block = info_->ep_n_host;
int num_he_block = info_->rp_n_host;
int num_ngd_block = (info_->ng_n_host + NGD_BLOCK_WARP_SIZE - 1) / NGD_BLOCK_WARP_SIZE;
k_29_fop_0<<<num_he_block, HE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, e.data_ptr<float>(), h.data_ptr<float>(), lr.data_ptr<float>(), em.data_ptr<float>(), ho.data_ptr<float>(), es.data_ptr<float>(), f.data_ptr<float>(), fo.data_ptr<float>());
}
__global__ void k_30_fop_0(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* e, float* h) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *ep_n) return;
row_panel info_tmp = ep_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (info_tmp.col_st == -1){ 
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
int row_tmp = info_tmp.row_st;
for (int nnz_tmp = col_st + threadIdx.x; nnz_tmp < col_ed; nnz_tmp += blockDim.x) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];
}
}
else {
int row_tmp = info_tmp.row_st;
for (int nnz_tmp = info_tmp.col_st + threadIdx.x; nnz_tmp < info_tmp.col_ed; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];}
}
}
__global__ void k_30_fop_1(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* h, float* lr, float* em) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *ep_n) return;
row_panel info_tmp = ep_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (info_tmp.col_st == -1){ 
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
using WarpReduce = cub::WarpReduce<float>;
__shared__ typename WarpReduce::TempStorage temp_storage[E_BLOCK_WARP_SIZE];
int row_tmp = info_tmp.row_st;
for (int col_iter = col_st; col_iter < col_ed; col_iter += blockDim.x) {
int nnz_tmp = col_iter + threadIdx.x;
int col_tmp = ColIdx[nnz_tmp];
int new_line_flag = 0; 
float res_tmp = 0;
if (nnz_tmp < col_ed) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
new_line_flag = (RowPtr[row_tmp] == nnz_tmp || lane_id == 0);
res_tmp = kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]);
}
else
res_tmp = -99999;
float reduce_tmp = WarpReduce(temp_storage[local_wid]).HeadSegmentedReduce(res_tmp, new_line_flag, cub::Max());
if (new_line_flag) atomicMax_float(&em[row_tmp * 1 + 0], reduce_tmp);
__syncwarp();
}
}
else {
int row_tmp = info_tmp.row_st;
float res_tmp = -99999;
for (int nnz_tmp = info_tmp.col_st + threadIdx.x; nnz_tmp < info_tmp.col_ed; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) {
atomicMax_float(&em[row_tmp * 1 + 0], res_tmp);
}
}
}
__global__ void k_30_fop_2(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* h, float* em, float* ho) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *ep_n) return;
row_panel info_tmp = ep_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (info_tmp.col_st == -1){ 
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
int row_tmp = info_tmp.row_st;
for (int nnz_tmp = col_st + threadIdx.x; nnz_tmp < col_ed; nnz_tmp += blockDim.x) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);
}
}
else {
int row_tmp = info_tmp.row_st;
for (int nnz_tmp = info_tmp.col_st + threadIdx.x; nnz_tmp < info_tmp.col_ed; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);}
}
}
__global__ void k_30_fop_3(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* ho, float* es) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *ep_n) return;
row_panel info_tmp = ep_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (info_tmp.col_st == -1){ 
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
using WarpReduce = cub::WarpReduce<float>;
__shared__ typename WarpReduce::TempStorage temp_storage[E_BLOCK_WARP_SIZE];
int row_tmp = info_tmp.row_st;
for (int col_iter = col_st; col_iter < col_ed; col_iter += blockDim.x) {
int nnz_tmp = col_iter + threadIdx.x;
int col_tmp = ColIdx[nnz_tmp];
int new_line_flag = 0; 
float res_tmp = 0;
if (nnz_tmp < col_ed) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
new_line_flag = (RowPtr[row_tmp] == nnz_tmp || lane_id == 0);
res_tmp = ho[nnz_tmp * 1 + 0];
}
float reduce_tmp = WarpReduce(temp_storage[local_wid]).HeadSegmentedReduce(res_tmp, new_line_flag, cub::Sum());
if (new_line_flag) atomicAdd(&es[row_tmp * 1 + 0], reduce_tmp);
__syncwarp();
}
}
else {
int row_tmp = info_tmp.row_st;
float res_tmp = 0;
for (int nnz_tmp = info_tmp.col_st + threadIdx.x; nnz_tmp < info_tmp.col_ed; nnz_tmp += HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) {
atomicAdd(&es[row_tmp * 1 + 0], res_tmp);
}
}
}
__global__ void k_30_fop_4(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* ho, float* f, float* es, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (global_wid >= *ng_n) return;
neighbor_group ng_tmp = ng_list[global_wid];
int row_tmp = ng_tmp.row_st;
int col_st = ng_tmp.col_st;
int col_ed = ng_tmp.col_st + NG_SIZE;
if (col_ed >= RowPtr[row_tmp + 1]) col_ed = RowPtr[row_tmp + 1];
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = col_st; nnz_tmp < col_ed; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
atomicAdd(&fo[row_tmp * featlen + k_tmp], res_tmp);
}
}
void gat_kernel_30(int64_t info_addr, torch::Tensor RowPtr, torch::Tensor ColIdx, int64_t featlen, torch::Tensor f, torch::Tensor fo, torch::Tensor we, torch::Tensor lr, torch::Tensor e, torch::Tensor em, torch::Tensor es, torch::Tensor h, torch::Tensor ho) {
kg_info* info_ = (kg_info*)info_addr;
if (info_ == nullptr) {
  throw std::runtime_error("info_ is null in gat_kernel_30");
}
if (info_->rp_info == nullptr || info_->rp_n == nullptr || info_->ep_info == nullptr || info_->ep_n == nullptr || info_->ng_info == nullptr || info_->ng_n == nullptr) {
  throw std::runtime_error("Some preprocessing pointers are null in gat_kernel_30");
}
int numnodes = RowPtr.size(0) - 1;
int numedges = ColIdx.size(0);
int num_n_block = (numnodes + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
int num_ne_block = (numnodes + NE_BLOCK_WARP_SIZE - 1) / NE_BLOCK_WARP_SIZE;
int num_e_block = info_->ep_n_host;
int num_he_block = info_->rp_n_host;
int num_ngd_block = (info_->ng_n_host + NGD_BLOCK_WARP_SIZE - 1) / NGD_BLOCK_WARP_SIZE;
k_30_fop_0<<<num_e_block, E_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, e.data_ptr<float>(), h.data_ptr<float>());
k_30_fop_1<<<num_e_block, E_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, h.data_ptr<float>(), lr.data_ptr<float>(), em.data_ptr<float>());
k_30_fop_2<<<num_e_block, E_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, h.data_ptr<float>(), em.data_ptr<float>(), ho.data_ptr<float>());
k_30_fop_3<<<num_e_block, E_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, ho.data_ptr<float>(), es.data_ptr<float>());
k_30_fop_4<<<num_ngd_block, NGD_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, ho.data_ptr<float>(), f.data_ptr<float>(), es.data_ptr<float>(), fo.data_ptr<float>());
}
__global__ void k31_fop0_dp0(int *RowPtr, int *ColIdx, int row_tmp, int featlen, float* e, float* h) {
int lane_id = threadIdx.x % WARP_SIZE;
int idx_st = RowPtr[row_tmp] + blockIdx.x * DP_NNZ_PER_BLOCK + threadIdx.x / WARP_SIZE * DP_NNZ_PER_WARP;
int idx_ed = idx_st + DP_NNZ_PER_WARP;
if (idx_ed > RowPtr[row_tmp + 1]) idx_ed = RowPtr[row_tmp + 1];
for (int nnz_tmp = idx_st + lane_id; nnz_tmp < idx_ed; nnz_tmp += WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];}
}
__global__ void k31_fop0_dp2(int *RowPtr, int *ColIdx, int row_tmp, int featlen, float* h, float* em, float* ho, float* es) {
int lane_id = threadIdx.x % WARP_SIZE;
int idx_st = RowPtr[row_tmp] + blockIdx.x * DP_NNZ_PER_BLOCK + threadIdx.x / WARP_SIZE * DP_NNZ_PER_WARP;
int idx_ed = idx_st + DP_NNZ_PER_WARP;
if (idx_ed > RowPtr[row_tmp + 1]) idx_ed = RowPtr[row_tmp + 1];
for (int nnz_tmp = idx_st + lane_id; nnz_tmp < idx_ed; nnz_tmp += WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);}
float res_tmp = 0;
for (int nnz_tmp = idx_st + lane_id; nnz_tmp < idx_ed; nnz_tmp+=WARP_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0];}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp += comm_tmp;
}
if (lane_id == 0) {
atomicAdd(&es[row_tmp * 1 + 0], res_tmp);
}
}
__global__ void k_31_fop_0(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* e, float* h, float* lr, float* em, float* ho, float* es) {
int lane_id = threadIdx.x % WARP_SIZE;
if (blockIdx.x >= *info_n) return;
row_panel info_tmp = info_list[blockIdx.x];
int local_wid = threadIdx.x / WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
int rows = info_tmp.row_ed - info_tmp.row_st;
int col_st = RowPtr[info_tmp.row_st];
int col_ed = RowPtr[info_tmp.row_ed];
int nnzs = col_ed - col_st;
if (info_tmp.col_st == -1) {
{ { int row_tmp = info_tmp.row_st;
for (int nnz_tmp = col_st + threadIdx.x; nnz_tmp < col_ed; nnz_tmp += HE_BLOCK_SIZE) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
int col_tmp = ColIdx[nnz_tmp];
h[nnz_tmp * 1 + 0] = e[row_tmp * 2 + 0]+e[col_tmp * 2 + 1];
}
}
}
__syncthreads();
{ using WarpReduce = cub::WarpReduce<float>;
__shared__ typename WarpReduce::TempStorage temp_storage[HE_BLOCK_WARP_SIZE];
int row_tmp = info_tmp.row_st;
for (int col_iter = col_st; col_iter < col_ed; col_iter += blockDim.x) {
int nnz_tmp = col_iter + threadIdx.x;
int col_tmp = ColIdx[nnz_tmp];
int new_line_flag = 0; 
float res_tmp = 0;
if (nnz_tmp < col_ed) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
new_line_flag = (RowPtr[row_tmp] == nnz_tmp || lane_id == 0);
res_tmp = kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]);
}
else
res_tmp = -99999;
float reduce_tmp = WarpReduce(temp_storage[local_wid]).HeadSegmentedReduce(res_tmp, new_line_flag, cub::Max());
if (new_line_flag) atomicMax_float(&em[row_tmp * 1 + 0], reduce_tmp);
__syncwarp();
}
}
__syncthreads();
{ { int row_tmp = info_tmp.row_st;
for (int nnz_tmp = col_st + threadIdx.x; nnz_tmp < col_ed; nnz_tmp += HE_BLOCK_SIZE) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
int col_tmp = ColIdx[nnz_tmp];
ho[nnz_tmp * 1 + 0] = expf(h[nnz_tmp * 1 + 0]-em[row_tmp * 1 + 0]);
}
}
}
__syncthreads();
{ using WarpReduce = cub::WarpReduce<float>;
__shared__ typename WarpReduce::TempStorage temp_storage[HE_BLOCK_WARP_SIZE];
int row_tmp = info_tmp.row_st;
for (int col_iter = col_st; col_iter < col_ed; col_iter += blockDim.x) {
int nnz_tmp = col_iter + threadIdx.x;
int col_tmp = ColIdx[nnz_tmp];
int new_line_flag = 0; 
float res_tmp = 0;
if (nnz_tmp < col_ed) {
while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;
new_line_flag = (RowPtr[row_tmp] == nnz_tmp || lane_id == 0);
res_tmp = ho[nnz_tmp * 1 + 0];
}
float reduce_tmp = WarpReduce(temp_storage[local_wid]).HeadSegmentedReduce(res_tmp, new_line_flag, cub::Sum());
if (new_line_flag) atomicAdd(&es[row_tmp * 1 + 0], reduce_tmp);
__syncwarp();
}
}
__syncthreads();
}
else {
int row_tmp = info_tmp.row_st;
int nnzs = RowPtr[row_tmp + 1] - RowPtr[row_tmp];
int dynamic_blocks = (nnzs + DP_NNZ_PER_BLOCK - 1) / DP_NNZ_PER_BLOCK;
if (threadIdx.x == 0) k31_fop0_dp0<<<dynamic_blocks, DP_BLOCK_SIZE>>>(RowPtr, ColIdx, row_tmp, featlen, e, h);
{
float res_tmp = -99999;
for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=HE_BLOCK_SIZE) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp = (kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]) > res_tmp)? kg_max(h[nnz_tmp * 1 + 0]*lr[0 * 1 + 0],h[nnz_tmp * 1 + 0]): res_tmp;}
for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {
float comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);
res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;
}
if (lane_id == 0) {
atomicMax_float(&em[row_tmp * 1 + 0], res_tmp);
}
}
if (threadIdx.x == 0) k31_fop0_dp2<<<dynamic_blocks, DP_BLOCK_SIZE>>>(RowPtr, ColIdx, row_tmp, featlen, h, em, ho, es);
}
}
__global__ void k_31_fop_1(int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n, float* ho, float* f, float* es, float* fo) {
int lane_id = threadIdx.x % WARP_SIZE;
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
int global_wid = global_tid / WARP_SIZE;
if (global_wid >= *ng_n) return;
neighbor_group ng_tmp = ng_list[global_wid];
int row_tmp = ng_tmp.row_st;
int col_st = ng_tmp.col_st;
int col_ed = ng_tmp.col_st + NG_SIZE;
if (col_ed >= RowPtr[row_tmp + 1]) col_ed = RowPtr[row_tmp + 1];
for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {
float res_tmp = 0;
for (int nnz_tmp = col_st; nnz_tmp < col_ed; nnz_tmp++) {
int col_tmp = ColIdx[nnz_tmp];
res_tmp += ho[nnz_tmp * 1 + 0]*f[col_tmp * featlen + k_tmp];}
res_tmp = res_tmp /es[row_tmp * 1 + 0];
atomicAdd(&fo[row_tmp * featlen + k_tmp], res_tmp);
}
}
void gat_kernel_31(int64_t info_addr, torch::Tensor RowPtr, torch::Tensor ColIdx, int64_t featlen, torch::Tensor f, torch::Tensor fo, torch::Tensor we, torch::Tensor lr, torch::Tensor e, torch::Tensor em, torch::Tensor es, torch::Tensor h, torch::Tensor ho) {
kg_info* info_ = (kg_info*)info_addr;
if (info_ == nullptr) {
  throw std::runtime_error("info_ is null in gat_kernel_31");
}
if (info_->rp_info == nullptr || info_->rp_n == nullptr || info_->ep_info == nullptr || info_->ep_n == nullptr || info_->ng_info == nullptr || info_->ng_n == nullptr) {
  throw std::runtime_error("Some preprocessing pointers are null in gat_kernel_31");
}
int numnodes = RowPtr.size(0) - 1;
int numedges = ColIdx.size(0);
int num_n_block = (numnodes + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;
int num_ne_block = (numnodes + NE_BLOCK_WARP_SIZE - 1) / NE_BLOCK_WARP_SIZE;
int num_e_block = info_->ep_n_host;
int num_he_block = info_->rp_n_host;
int num_ngd_block = (info_->ng_n_host + NGD_BLOCK_WARP_SIZE - 1) / NGD_BLOCK_WARP_SIZE;
k_31_fop_0<<<num_he_block, HE_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, e.data_ptr<float>(), h.data_ptr<float>(), lr.data_ptr<float>(), em.data_ptr<float>(), ho.data_ptr<float>(), es.data_ptr<float>());
k_31_fop_1<<<num_ngd_block, NGD_BLOCK_SIZE>>>(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), featlen, info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n, ho.data_ptr<float>(), f.data_ptr<float>(), es.data_ptr<float>(), fo.data_ptr<float>());
}
TORCH_LIBRARY(gatlib, m) {
m.def("gat_kernel_0", &gat_kernel_0);
m.def("gat_kernel_1", &gat_kernel_1);
m.def("gat_kernel_2", &gat_kernel_2);
m.def("gat_kernel_3", &gat_kernel_3);
m.def("gat_kernel_4", &gat_kernel_4);
m.def("gat_kernel_5", &gat_kernel_5);
m.def("gat_kernel_6", &gat_kernel_6);
m.def("gat_kernel_7", &gat_kernel_7);
m.def("gat_kernel_8", &gat_kernel_8);
m.def("gat_kernel_9", &gat_kernel_9);
m.def("gat_kernel_10", &gat_kernel_10);
m.def("gat_kernel_11", &gat_kernel_11);
m.def("gat_kernel_12", &gat_kernel_12);
m.def("gat_kernel_13", &gat_kernel_13);
m.def("gat_kernel_14", &gat_kernel_14);
m.def("gat_kernel_15", &gat_kernel_15);
m.def("gat_kernel_16", &gat_kernel_16);
m.def("gat_kernel_17", &gat_kernel_17);
m.def("gat_kernel_18", &gat_kernel_18);
m.def("gat_kernel_19", &gat_kernel_19);
m.def("gat_kernel_20", &gat_kernel_20);
m.def("gat_kernel_21", &gat_kernel_21);
m.def("gat_kernel_22", &gat_kernel_22);
m.def("gat_kernel_23", &gat_kernel_23);
m.def("gat_kernel_24", &gat_kernel_24);
m.def("gat_kernel_25", &gat_kernel_25);
m.def("gat_kernel_26", &gat_kernel_26);
m.def("gat_kernel_27", &gat_kernel_27);
m.def("gat_kernel_28", &gat_kernel_28);
m.def("gat_kernel_29", &gat_kernel_29);
m.def("gat_kernel_30", &gat_kernel_30);
m.def("gat_kernel_31", &gat_kernel_31);
m.def("preprocessing", &preprocessing);
}