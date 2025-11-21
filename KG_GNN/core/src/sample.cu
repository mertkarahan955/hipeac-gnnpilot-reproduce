__shared__ int A_shared[32][32];
__shared__ int B_shared[32][32];
__shared__ int C_shared[32][32];

for (int k = 0; k < N; k+=32)
{
    A_shared[threadIdx.x][threadIdx.y] = A[row * N + k + threadIdx.x];
    B_shared[threadIdx.x][threadIdx.y] = B[(k + threadIdx.y) * N + col];
    C_shared[threadIdx.x] = 0;

    __syncthreads();

    for (int kk = 0; kk < 32; kk++)
        C_shared[threadIdx.x][threadIdx.y] += A_shared[threadIdx.x][kk] * A_shared[kk][threadIdx.y];

    C[row * N + col] = C_shared[threadIdx.x][threadIdx.y];
}


