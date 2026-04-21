#ifndef CSR_SPVM
#define CSR_SPVM
#include "coo_to_csr.h"
void spmv_csr(const CSR_Matrix *A, const float *x, float *y);
#endif