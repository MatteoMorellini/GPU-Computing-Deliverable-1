#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "mtx_reader.h"
#include "coo_to_csr.h"
#include "generate_dense.h"
#include "csr_spvm.h"
#include "time_lib.h"
#include "perf_stats.h"

#define REPS 100
#define WARMUP 5


int main(void) {
    srand(0); // set seed for reproducibility

    DIR *d;
    struct dirent *dir;
    char folder[] = "./dummy/";
    char path[1024];

    d = opendir(folder);
    if (d == NULL) {
        printf("Error opening directory\n");
        return 1;
    }

    printf("Files in matrices directory:\n");

    TIMER_DEF(0);

    while ((dir = readdir(d)) != NULL) {
        if (dir->d_name[0] == '.')
            continue;

        char *ext = strrchr(dir->d_name, '.');
        if (!ext || strcmp(ext, ".mtx") != 0)
            continue;

        // -------------------------------------------------------
        // MATRIX LOADING AND CONVERSION SECTION

        snprintf(path, sizeof(path), "%s%s", folder, dir->d_name);

        COO_Matrix A;
        read_mtx(path, &A);
        printf("Read matrix %s: %d rows, %d cols, %d non-zeros\n",
               dir->d_name, A.rows, A.cols, A.nnz);

        CSR_Matrix csr_A;
        coo_to_csr(&A, &csr_A);

        PerfStats stats;
        memset(&stats, 0, sizeof(stats));

        strncpy(stats.name, dir->d_name, sizeof(stats.name) - 1);
        strncpy(stats.format, "CSR", sizeof(stats.format) - 1);
        strncpy(stats.implementation, "CPU Single-Core", sizeof(stats.implementation) - 1);

        stats.rows  = csr_A.rows;
        stats.cols  = csr_A.cols;
        stats.nnz   = csr_A.nnz;
        stats.valid = 1; // assume valid until checked otherwise

        // -------------------------------------------------------
        // PREPARATION SECTION

        float *x = malloc((size_t)csr_A.cols * sizeof(*x));
        if (x == NULL) {
            fprintf(stderr, "Error allocating memory for dense vector\n");
            free_coo(&A);
            free_csr(&csr_A);
            closedir(d);
            return 1;
        }
        fill_dense(x, (size_t)csr_A.cols);

        float *y = malloc((size_t)csr_A.rows * sizeof(*y));
        if (y == NULL) {
            fprintf(stderr, "Error: could not allocate output vector y\n");
            free_coo(&A);
            free_csr(&csr_A);
            free_dense(x);
            closedir(d);
            return 1;
        }

        // -------------------------------------------------------
        // WARMUP + TIMING SECTION

        double times[REPS];

        for (int r = 0; r < WARMUP; r++)
            spmv_csr(&csr_A, x, y);

        for (int r = 0; r < REPS; r++) {
            TIMER_START(0);
            spmv_csr(&csr_A, x, y);
            TIMER_STOP(0);
            times[r] = TIMER_ELAPSED(0) / 1e6; // microseconds -> seconds
        }

        // -------------------------------------------------------
        // PERFORMANCE METRICS SECTION

        double total_time = 0.0;
        for (int r = 0; r < REPS; r++)
            total_time += times[r];

        double avg_time = total_time / REPS;
        double gflops   = (2.0 * csr_A.nnz) / (avg_time * 1e9);

        double variance = 0.0;
        for (int r = 0; r < REPS; r++) {
            double diff = times[r] - avg_time;
            variance += diff * diff;
        }
        variance /= REPS;
        double std_time = sqrt(variance);

        stats.avg_time_s = avg_time;
        stats.std_time_s = std_time;
        stats.gflops     = gflops;

        // -------------------------------------------------------
        // OUTPUT SECTION

        printf("Average time for %s: %.9f s\n",            dir->d_name, stats.avg_time_s);
        printf("GFLOP/s for %s: %.6f\n",                   dir->d_name, stats.gflops);
        printf("Standard deviation of time for %s: %.9f s\n", dir->d_name, stats.std_time_s);
        printf("\n");

        free_coo(&A);
        free_csr(&csr_A);
        free_dense(x);
        free(y);
    }

    closedir(d);
    return 0;
}