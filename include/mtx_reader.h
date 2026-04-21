#ifndef MTX_READER_H
#define MTX_READER_H

typedef struct {
    int rows;
    int cols;
    int nnz;
    int *row;
    int *col;
    double *data;
} COO_Matrix;

COO_Matrix read_mtx(const char *filename);
void free_coo_matrix(COO_Matrix *mat);

#endif