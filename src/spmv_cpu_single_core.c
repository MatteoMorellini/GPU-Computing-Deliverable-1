#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include "mtx_reader.h"
#include "coo_to_csr.h"
#include "generate_dense.h"
#include "csr_spvm.h"

int main(){

    DIR *d;
    struct dirent *dir;
    char folder[] = "./matrices/";
    char path[1024];

    d = opendir(folder);
    if(d == NULL){
        printf("Error opening directory\n");
        return 1;
    }

    printf("Files in matrices directory:\n");

    while ((dir = readdir(d)) != NULL){
        if (dir->d_name[0] == '.') continue;

        snprintf(path, sizeof(path), "%s%s", folder, dir->d_name);
        COO_Matrix A = read_mtx(path);
        CSR_Matrix csr_A = coo_to_csr(A);
        
        printf("Matrix %s: %d rows, %d cols, %d non-zeros\n", dir->d_name, csr_A.rows, csr_A.cols, csr_A.nnz);
        float* x = generate_dense(csr_A.cols);
        /*for(int i = 0; i < csr_A.cols; i++){
            printf("x[%d] = %f\n", i, x[i]);
        }*/

        float y[csr_A.rows];
        spmv_csr(&csr_A, x, y);
        /*
        for(int i = 0; i < csr_A.rows; i++){
            printf("y[%d] = %f\n", i, y[i]);
        }*/

        free_coo(&A);
        free_csr(&csr_A);
        free_dense(x);
    }
    closedir(d);
    return 0;
}