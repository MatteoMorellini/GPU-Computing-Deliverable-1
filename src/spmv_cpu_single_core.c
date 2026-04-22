#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <time.h>
#include "mtx_reader.h"
#include "coo_to_csr.h"
#include "generate_dense.h"
#include "csr_spvm.h"
#include "time_lib.h"
#define REPS 100
#define WARMUP 5

typedef struct {
    char   name[256];
    char format[32];
    char implementation[32];

    int    rows;
    int    cols;
    int    nnz;

    double avg_nnz_per_row;
    double std_nnz_per_row;

    double avg_time_s;
    double std_time_s;

    double gflops;
    int valid; // boolean flag to indicate if the matrix was processed successfully
    double max_abs_error; // maximum absolute error compared to a reference implementation (CPU)
} MatrixStats;

int main(){
    srand(0); // set seed for reproducibility
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

    TIMER_DEF(0);
    while ((dir = readdir(d)) != NULL){
        if (dir->d_name[0] == '.') continue;

        // -------------------------------------------------------
        // MATRIX LOADING AND CONVERSION SECTION

        snprintf(path, sizeof(path), "%s%s", folder, dir->d_name);
        COO_Matrix A;
        read_mtx(path, &A); 
        printf("Read matrix %s: %d rows, %d cols, %d non-zeros\n", dir->d_name,\
            A.rows, A.cols, A.nnz);
        CSR_Matrix csr_A;
        coo_to_csr(&A, &csr_A);
            
        // -------------------------------------------------------
        // PREPARATION SECTION

        float *x = malloc((size_t)csr_A.cols * sizeof(*x));
        if (x == NULL) {
            fprintf(stderr, "Error allocating memory for dense vector\n");
            return 1;
        }
        fill_dense(x, (size_t)csr_A.cols);

        float *y = malloc(csr_A.rows * sizeof(float));
        if (y == NULL) {
            fprintf(stderr, "Error: could not allocate output vector y\n");
            return 1;
        }

        // -------------------------------------------------------
        // WARMUP + TIMING SECTION

        // timer excludes .mtx parsing, COO to CSR conversion, and dense vector
        for (int r = 0; r < WARMUP; r++)
            spmv_csr(&csr_A, x, y);  

        TIMER_START(0);
        for (int r = 0; r < REPS; r++)
            spmv_csr(&csr_A, x, y);
        TIMER_STOP(0);

        // -------------------------------------------------------
        // METRICS SECTION
         
        double total_time = TIMER_ELAPSED(0) / 1e6;   // convert to seconds
        double avg_time = total_time / REPS;
        double gflops = (2.0 * csr_A.nnz) / (avg_time * 1e9);

        printf("Average time for %s: %.9f s\n", dir->d_name, avg_time);
        printf("GFLOP/s for %s: %.6f\n", dir->d_name, gflops);

        free_coo(&A);
        free_csr(&csr_A);
        free_dense(x);
        free(y);
    }

    // todo: use geometric mean for FLOP/s across all matrices?

    closedir(d);
    return 0;
}