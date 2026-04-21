#include <stdio.h>
#include <stdlib.h>
#include "mtx_reader.h"
 
COO_Matrix read_mtx(const char *filename) {
    FILE *f;
    char line[1024];
    COO_Matrix mat;
    f = fopen(filename, "r");
    if(!f){
        printf("Error opening file %s\n", filename);
        exit(1);
    }
    do {
        fgets(line, 1024, f);
    }while(line[0] == '%');
    // in .mtx files, lines beginning with % are comments/metadata
    // the first line after them contains matrix dimensions 
    // then the following lines contain the non-zero entries in the 
    // row col value
        
    sscanf(line, "%d %d %d", &mat.rows, &mat.cols, &mat.nnz);
    mat.row = malloc(mat.nnz * sizeof(int));
    mat.col = malloc(mat.nnz * sizeof(int));
    mat.data = malloc(mat.nnz * sizeof(double)); 
    // TODO: #1 convert from double to int
    // at this point the pointer is positioned at the 1st matrix entry line
    for(int i=0; i<mat.nnz; i++){
        fscanf(f, "%d %d %lf", &mat.row[i], &mat.col[i], &mat.data[i]);
        // convert to 0-based indexing
        mat.row[i]--;
        mat.col[i]--;
    }
    fclose(f);
    return mat;
}

void free_coo(COO_Matrix *mat){
    free(mat->row);
    free(mat->col);
    free(mat->data);
}