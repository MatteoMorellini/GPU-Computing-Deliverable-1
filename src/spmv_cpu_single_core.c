#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <time.h>
#include "mtx_reader.h"
#include "coo_to_csr.h"
#include "generate_dense.h"
#include "csr_spvm.h"
#include "time_lib.h"

int main(){

    DIR *d;
    struct dirent *dir;
    char folder[] = "./matrices/";
    char path[1024];
    double timers[10];

    d = opendir(folder);
    if(d == NULL){
        printf("Error opening directory\n");
        return 1;
    }

    printf("Files in matrices directory:\n");

    TIMER_DEF(0);
    int i = 0;
    while ((dir = readdir(d)) != NULL){
        if (dir->d_name[0] == '.') continue;
        printf("%s\n", dir->d_name);
        snprintf(path, sizeof(path), "%s%s", folder, dir->d_name);
        COO_Matrix A = read_mtx(path);
        printf("Read matrix %s: %d rows, %d cols, %d non-zeros\n", dir->d_name, A.rows, A.cols, A.nnz);
        CSR_Matrix csr_A = coo_to_csr(A);

        printf("Matrix %s: %d rows, %d cols, %d non-zeros\n", dir->d_name, csr_A.rows, csr_A.cols, csr_A.nnz);
        float* x = generate_dense(csr_A.cols);

        float *y = malloc(csr_A.rows * sizeof(float));
        if (y == NULL) {
            fprintf(stderr, "Error: could not allocate output vector y\n");
            return 1;
        }

        TIMER_START(0);
        spmv_csr(&csr_A, x, y);
        TIMER_STOP(0);
        timers[i] = TIMER_ELAPSED(0) / 1e3; // convert to milliseconds 
        printf("Time taken for %s: %lf ms\n", dir->d_name, timers[i]);
        i++;
        /*
        for(int i = 0; i < csr_A.rows; i++){
            printf("y[%d] = %f\n", i, y[i]);
        }*/

        //TODO: don't return values from functions but modify them in-place to avoid memory leaks
        free_coo(&A);
        free_csr(&csr_A);
        free_dense(x);
    }
    closedir(d);
    return 0;
}