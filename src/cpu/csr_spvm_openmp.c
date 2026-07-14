#include "csr_spvm.h"

void spmv_csr_openmp(const CSR_Matrix *A, const float *x, float *y)
{
    /* CSR rows are independent, so each thread owns a disjoint y element. */
#pragma omp parallel for schedule(static)
    for (int row = 0; row < A->rows; row++) {
        float sum = 0.0f;

        for (int entry = A->row_ptr[row]; entry < A->row_ptr[row + 1]; entry++) {
            sum += A->values[entry] * x[A->col_idx[entry]];
        }

        y[row] = sum;
    }
}

void spmv_csr_reference_openmp(const CSR_Matrix *A, const float *x, double *y)
{
    /*
     * Accumulating in double makes this a more accurate correctness oracle
     * for GPU kernels while retaining row-level OpenMP parallelism. Each
     * thread still owns a distinct output element, so no synchronization is
     * needed inside the sparse dot product.
     */
#pragma omp parallel for schedule(static)
    for (int row = 0; row < A->rows; row++) {
        double sum = 0.0;

        for (int entry = A->row_ptr[row]; entry < A->row_ptr[row + 1]; entry++) {
            sum += (double)A->values[entry] * (double)x[A->col_idx[entry]];
        }

        y[row] = sum;
    }
}
