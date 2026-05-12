#include <stdio.h>
#include <stdlib.h>
#include "csr_spvm.h"
#include "coo_to_csr.h"

void spmv_csr(const CSR_Matrix *A, const float *x, float *y){
    for (int i = 0; i < A->rows; i++){
        y[i] = 0.0f;
        for (int j = A->row_ptr[i]; j < A->row_ptr[i+1]; j++){
            y[i] += A->values[j] * x[A->col_idx[j]];
        }
    }
}
