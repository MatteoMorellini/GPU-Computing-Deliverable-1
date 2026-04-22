#ifndef COO_TO_CSR_H
#define COO_TO_CSR_H

#include "mtx_reader.h"
typedef struct{
    int rows;
    int cols;
    int nnz;
    int *row_ptr;
    int *col_idx;
    float *values;
} CSR_Matrix;

void coo_to_csr(COO_Matrix *coo, CSR_Matrix *csr);
void free_csr(CSR_Matrix *csr);

#endif