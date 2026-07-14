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
