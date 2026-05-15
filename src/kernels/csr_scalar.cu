#include "gpu_bench.cuh"

__global__ void csr_scalar_kernel(CSR_Matrix mat,
                                  const float * __restrict__ x,
                                  float * __restrict__ y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < mat.rows) {
        float sum = 0.0f;
        for (int j = mat.row_ptr[row]; j < mat.row_ptr[row + 1]; j++)
            sum += mat.values[j] * x[mat.col_idx[j]];
        y[row] = sum;
    }
}

struct CsrScalarImpl : NoPrepImpl {
    void launch(const CSR_Matrix &d_csr, const float *d_x, float *d_y) {
        int threads = 256;
        int blocks  = (d_csr.rows + threads - 1) / threads;
        csr_scalar_kernel<<<blocks, threads>>>(d_csr, d_x, d_y);
    }
};

int main(void) {
    srand(0);
    CsrScalarImpl impl;
    BenchConfig cfg{"CUDA CSR-Scalar", "CSR", "results/gpu_csr_scalar.csv"};
    return run_benchmark(cfg, impl);
}
