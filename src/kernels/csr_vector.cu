#include "gpu_bench.cuh"

// One warp per row: 32 lanes stride through the row's nonzeros, then a
// shuffle-based warp reduction writes the result.
__global__ void csr_vector_kernel(const CSR_Matrix mat,
                                  const float * __restrict__ x,
                                  float * __restrict__ y) {
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    int row  = global_thread_id / 32;

    if (row < mat.rows) {
        int row_start = mat.row_ptr[row];
        int row_end   = mat.row_ptr[row + 1];

        float sum = 0.0f;
        for (int j = row_start + lane; j < row_end; j += 32)
            sum += mat.values[j] * x[mat.col_idx[j]];

        for (int offset = 16; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, offset);

        if (lane == 0) y[row] = sum;
    }
}

struct CsrVectorImpl : NoPrepImpl {
    void launch(const CSR_Matrix &d_csr, const float *d_x, float *d_y) {
        int threads_per_block = 256;         
        int warps_per_block   = threads_per_block / 32;
        int blocks = (d_csr.rows + warps_per_block - 1) / warps_per_block;
        csr_vector_kernel<<<blocks, threads_per_block>>>(d_csr, d_x, d_y);
    }
};

int main(void) {
    srand(0);
    CsrVectorImpl impl;
    BenchConfig cfg{"CUDA CSR-Vector", "CSR", "results/gpu_csr_vector.csv"};
    return run_benchmark(cfg, impl);
}
