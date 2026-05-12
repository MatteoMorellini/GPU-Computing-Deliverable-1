#include <stdio.h>
// we import mtx_reader.h to guarantee that the implementation matches 
// the declared interface
#include <stdlib.h>
#include "coo_to_csr.h"

void coo_to_csr(COO_Matrix *coo, CSR_Matrix *csr){
    csr->rows = coo->rows;
    csr->cols = coo->cols;
    csr->nnz = coo->nnz;
        
    // zero-initialize row_ptr 
    csr->row_ptr = calloc(coo->rows + 1, sizeof(int));
    csr->col_idx = malloc(csr->nnz * sizeof(int));
    csr->values = malloc(csr->nnz * sizeof(float));
    if (!csr->row_ptr || !csr->col_idx || !csr->values){
        printf("Error allocating memory for CSR matrix\n");
        exit(1);
    }

    for (int i = 0; i < coo->nnz; i++) {
        csr->row_ptr[coo->row[i] + 1]++;
    }

    for (int i = 0; i < csr->rows; i++) {
        csr->row_ptr[i + 1] += csr->row_ptr[i];
    }
        
    // rows might be out-of-order, so we need to keep track of the next 
    // available position for each row
    int *next = malloc(csr->rows * sizeof(int));
    for (int i = 0; i < csr->rows; i++) {
        next[i] = csr->row_ptr[i];
    }

    for (int i = 0; i < coo->nnz; i++) {
        int row = coo->row[i];
        int dest = next[row];

        csr->col_idx[dest] = coo->col[i];
        csr->values[dest]  = (float) coo->data[i];

        next[row]++;
    }

    free(next);
}

void free_csr(CSR_Matrix *csr){
    free(csr->row_ptr);
    free(csr->col_idx);
    free(csr->values);
}