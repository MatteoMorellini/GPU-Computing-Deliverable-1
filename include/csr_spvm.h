#ifndef CSR_SPVM
#define CSR_SPVM
#include "coo_to_csr.h"
void spmv_csr(const CSR_Matrix *A, const float *x, float *y);
void spmv_csr_openmp(const CSR_Matrix *A, const float *x, float *y);
void spmv_csr_reference_openmp(const CSR_Matrix *A, const float *x, double *y);
#endif
