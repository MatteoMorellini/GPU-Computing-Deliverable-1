#ifndef CSR_DEVICE_CUH
#define CSR_DEVICE_CUH

#include <cuda_runtime.h>
#include <stddef.h>

extern "C" {
#include "coo_to_csr.h"
}

// Allocate device storage for the CSR arrays and copy them from host.
static inline void csr_to_device(const CSR_Matrix *h_mat, CSR_Matrix *d_mat) {
    d_mat->rows = h_mat->rows;
    d_mat->cols = h_mat->cols;
    d_mat->nnz  = h_mat->nnz;

    cudaMalloc((void**)&d_mat->row_ptr, (size_t)(h_mat->rows + 1) * sizeof(int));
    cudaMalloc((void**)&d_mat->col_idx, (size_t)h_mat->nnz        * sizeof(int));
    cudaMalloc((void**)&d_mat->values,  (size_t)h_mat->nnz        * sizeof(float));

    cudaMemcpy(d_mat->row_ptr, h_mat->row_ptr,
               (size_t)(h_mat->rows + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat->col_idx, h_mat->col_idx,
               (size_t)h_mat->nnz * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat->values, h_mat->values,
               (size_t)h_mat->nnz * sizeof(float),
               cudaMemcpyHostToDevice);
}

static inline void csr_device_free(CSR_Matrix *d_mat) {
    cudaFree(d_mat->row_ptr);
    cudaFree(d_mat->col_idx);
    cudaFree(d_mat->values);
    d_mat->row_ptr = nullptr;
    d_mat->col_idx = nullptr;
    d_mat->values  = nullptr;
}

#endif // CSR_DEVICE_CUH
