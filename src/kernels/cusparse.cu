#include "gpu_bench.cuh"

#include <cusparse_v2.h>
#include <cuda.h>

#define CHECK_CUDA(call) do {                                          \
    cudaError_t err = (call);                                          \
    if (err != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while (0)

#define CHECK_CUSPARSE(call) do {                                      \
    cusparseStatus_t status = (call);                                  \
    if (status != CUSPARSE_STATUS_SUCCESS) {                           \
        fprintf(stderr, "cuSPARSE error at %s:%d: status %d\n",        \
                __FILE__, __LINE__, status);                           \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while (0)

struct CuSparseImpl {
    cusparseHandle_t      handle  = nullptr;
    cusparseSpMatDescr_t  matA    = nullptr;
    cusparseDnVecDescr_t  vecX    = nullptr;
    cusparseDnVecDescr_t  vecY    = nullptr;
    void                 *dBuffer = nullptr;
    float                 alpha   = 1.0f;
    float                 beta    = 0.0f;

    CuSparseImpl() {
        CHECK_CUSPARSE(cusparseCreate(&handle));
    }
    ~CuSparseImpl() {
        if (handle) cusparseDestroy(handle);
    }

    double prep(const CSR_Matrix &, const CSR_Matrix &d_csr,
                float *d_x, float *d_y) {
        TIMER_DEF(t);
        TIMER_START(t);

        CHECK_CUSPARSE(cusparseCreateCsr(
            &matA,
            d_csr.rows, d_csr.cols, d_csr.nnz,
            d_csr.row_ptr, d_csr.col_idx, d_csr.values,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

        CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, d_csr.cols, d_x, CUDA_R_32F));
        CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, d_csr.rows, d_y, CUDA_R_32F));

        size_t bufferSize = 0;
        CHECK_CUSPARSE(cusparseSpMV_bufferSize(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, vecX, &beta, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2, &bufferSize));

        CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

        TIMER_STOP(t);
        return TIMER_ELAPSED(t) / 1e6;
    }

    void launch(const CSR_Matrix &, const float *, float *) {
        CHECK_CUSPARSE(cusparseSpMV(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, vecX, &beta, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2, dBuffer));
    }

    void teardown() {
        if (matA)    { CHECK_CUSPARSE(cusparseDestroySpMat(matA)); matA = nullptr; }
        if (vecX)    { CHECK_CUSPARSE(cusparseDestroyDnVec(vecX)); vecX = nullptr; }
        if (vecY)    { CHECK_CUSPARSE(cusparseDestroyDnVec(vecY)); vecY = nullptr; }
        if (dBuffer) { CHECK_CUDA(cudaFree(dBuffer));              dBuffer = nullptr; }
    }
};

int main(void) {
    srand(0);
    CuSparseImpl impl;
    BenchConfig cfg{"GPU cuSPARSE", "CSR", "results/cusparse.csv"};
    return run_benchmark(cfg, impl);
}
