#include <stdio.h>
// we import mtx_reader.h to guarantee that the implementation matches 
// the declared interface
#include <stdlib.h>
#include "coo_to_csr.h"

CSR_Matrix coo_to_csr(COO_Matrix coo){
    CSR_Matrix csr;
    csr.rows = coo.rows;
    csr.cols = coo.cols;
    csr.nnz = coo.nnz;

    csr.row_ptr = malloc((csr.rows+1) * sizeof(int));
    csr.col_idx = malloc(csr.nnz * sizeof(int));
    csr.values = malloc(csr.nnz * sizeof(float));
    if (!csr.row_ptr || !csr.col_idx || !csr.values){
        printf("Error allocating memory for CSR matrix\n");
        exit(1);
    }

    csr.row_ptr[0] = 0;
    for(int i=0; i<coo.nnz; i++){
        csr.row_ptr[coo.row[i]+1]++;
    }
    return csr;
}