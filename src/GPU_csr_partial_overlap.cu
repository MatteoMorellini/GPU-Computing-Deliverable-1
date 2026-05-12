#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
extern "C" {
    #include "mtx_reader.h"
    #include "coo_to_csr.h"
    #include "generate_dense.h"
    #include "csr_spvm.h"
    #include "time_lib.h"
    #include "perf_stats.h"
}
#include <cusparse_v2.h>
#include <cuda.h>

#define REPS 100
#define WARMUP 5

#define CHECK_CUDA(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",              \
                __FILE__, __LINE__, cudaGetErrorString(err));      \
        exit(EXIT_FAILURE);                                        \
    }                                                             \
} while (0)

#define CHECK_CUSPARSE(call) do {                                  \
    cusparseStatus_t status = (call);                              \
    if (status != CUSPARSE_STATUS_SUCCESS) {                       \
        fprintf(stderr, "cuSPARSE error at %s:%d: status %d\n",   \
                __FILE__, __LINE__, status);                       \
        exit(EXIT_FAILURE);                                        \
    }                                                             \
} while (0)

void csr_to_device(const CSR_Matrix *h_mat, CSR_Matrix *d_mat) {
    d_mat->rows = h_mat->rows;
    d_mat->cols = h_mat->cols;
    d_mat->nnz  = h_mat->nnz;

    d_mat->row_ptr = NULL;
    d_mat->col_idx = NULL;
    d_mat->values  = NULL;

    CHECK_CUDA(cudaMalloc((void**)&d_mat->row_ptr,
                          (size_t)(h_mat->rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_mat->col_idx,
                          (size_t)h_mat->nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_mat->values,
                          (size_t)h_mat->nnz * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_mat->row_ptr, h_mat->row_ptr,
                          (size_t)(h_mat->rows + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mat->col_idx, h_mat->col_idx,
                          (size_t)h_mat->nnz * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mat->values, h_mat->values,
                          (size_t)h_mat->nnz * sizeof(float),
                          cudaMemcpyHostToDevice));
}

int main(void) {
    srand(0);

    // Create cuSPARSE handle once, reuse across all matrices
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    DIR *d;
    struct dirent *dir;
    char folder[] = "./matrices/";
    char path[1024];

    d = opendir(folder);
    if (d == NULL) {
        printf("Error opening directory\n");
        cusparseDestroy(handle);
        return 1;
    }

    printf("Files in matrices directory:\n");

    FILE *csv = perf_stats_open_csv(
        perf_stats_resolve_path("results/cusparse.csv"));
    if (!csv) { closedir(d); cusparseDestroy(handle); return 1; }

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

        strncpy(stats.name,           dir->d_name,    sizeof(stats.name) - 1);
        strncpy(stats.format,         "CSR",           sizeof(stats.format) - 1);
        strncpy(stats.implementation, "GPU cuSPARSE",  sizeof(stats.implementation) - 1);

        stats.rows  = csr_A.rows;
        stats.cols  = csr_A.cols;
        stats.nnz   = csr_A.nnz;
        stats.valid = 1;

        // -------------------------------------------------------
        // PREPARATION SECTION

        float *x = (float*)malloc((size_t)csr_A.cols * sizeof(*x));
        if (x == NULL) {
            fprintf(stderr, "Error allocating memory for dense vector\n");
            free_coo(&A); free_csr(&csr_A); closedir(d);
            cusparseDestroy(handle);
            return 1;
        }
        fill_dense(x, (size_t)csr_A.cols);

        float *y = (float*)malloc((size_t)csr_A.rows * sizeof(*y));
        if (y == NULL) {
            fprintf(stderr, "Error: could not allocate output vector y\n");
            free_coo(&A); free_csr(&csr_A); free_dense(x); closedir(d);
            cusparseDestroy(handle);
            return 1;
        }

        // -------------------------------------------------------
        // MOVE TO GPU

        float *d_x = NULL;
        float *d_y = NULL;
        CHECK_CUDA(cudaMalloc((void**)&d_x, (size_t)csr_A.cols * sizeof(float)));
        CHECK_CUDA(cudaMalloc((void**)&d_y, (size_t)csr_A.rows * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_x, x, (size_t)csr_A.cols * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_y, y, (size_t)csr_A.rows * sizeof(float), cudaMemcpyHostToDevice));

        CSR_Matrix d_csr_A;
        csr_to_device(&csr_A, &d_csr_A);

        // -------------------------------------------------------
        // CUSPARSE DESCRIPTORS

        cusparseSpMatDescr_t matA;
        cusparseDnVecDescr_t vecX, vecY;

        CHECK_CUSPARSE(cusparseCreateCsr(
            &matA,
            d_csr_A.rows, d_csr_A.cols, d_csr_A.nnz,
            d_csr_A.row_ptr,
            d_csr_A.col_idx,
            d_csr_A.values,
            CUSPARSE_INDEX_32I, //row_ptr type is int 32-bit
            CUSPARSE_INDEX_32I, // col_idx type is int 32-bit
            CUSPARSE_INDEX_BASE_ZERO, // zero-based indexing
            CUDA_R_32F // values type is float 32-bit real
        ));

        CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, d_csr_A.cols, d_x, CUDA_R_32F));
        CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, d_csr_A.rows, d_y, CUDA_R_32F));

        // Query and allocate external buffer for SpMV
        float alpha = 1.0f, beta = 0.0f;
        size_t bufferSize = 0;
        void  *dBuffer    = NULL;

        CHECK_CUSPARSE(cusparseSpMV_bufferSize(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, vecX,
            &beta,  vecY,
            CUDA_R_32F,
            CUSPARSE_SPMV_CSR_ALG2,
            &bufferSize
        ));

        CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

        // -------------------------------------------------------
        // WARMUP + TIMING SECTION

        double times[REPS];

        // Fixed: braces around warmup body so cudaDeviceSynchronize is inside the loop
        for (int r = 0; r < WARMUP; r++) {
            CHECK_CUSPARSE(cusparseSpMV(
                handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, vecX,
                &beta,  vecY,
                CUDA_R_32F,
                CUSPARSE_SPMV_CSR_ALG2,
                dBuffer
            ));
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        for (int r = 0; r < REPS; r++) {
            TIMER_START(0);
            CHECK_CUSPARSE(cusparseSpMV(
                handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, vecX,
                &beta,  vecY,
                CUDA_R_32F,
                CUSPARSE_SPMV_CSR_ALG2,
                dBuffer
            ));
            CHECK_CUDA(cudaDeviceSynchronize());
            TIMER_STOP(0);
            times[r] = TIMER_ELAPSED(0) / 1e6; // microseconds -> seconds
        }

        CHECK_CUDA(cudaMemcpy(y, d_y, (size_t)csr_A.rows * sizeof(float), cudaMemcpyDeviceToHost));
        
        // -------------------------------------------------------
        // CORRECTNESS CHECK: compare GPU result `y` with CPU `spmv_csr`
        {
            float *cpu_y = (float*)malloc((size_t)csr_A.rows * sizeof(float));
            if (cpu_y) {
                spmv_csr(&csr_A, x, cpu_y);
                int mismatches = 0;
                double max_abs_err = 0.0;
                const float tol = 1e-4f;
                for (int i = 0; i < csr_A.rows; i++) {
                    float a = y[i];
                    float b = cpu_y[i];
                    float abs_err = fabsf(a - b);
                    if (abs_err > tol * fmaxf(1.0f, fabsf(b))) {
                        if (mismatches < 5)
                            printf("  MISMATCH row %d: gpu=%g cpu=%g abs_err=%g\n", i, a, b, abs_err);
                        mismatches++;
                        if (abs_err > max_abs_err) max_abs_err = abs_err;
                    }
                }
                if (mismatches) {
                    printf("  CORRECTNESS: %d mismatches (max_abs_err=%g)\n", mismatches, max_abs_err);
                    stats.valid = 0;
                } else {
                    printf("  CORRECTNESS: OK\n");
                }
                stats.max_abs_error = max_abs_err;
                free(cpu_y);
            } else {
                fprintf(stderr, "Warning: could not allocate cpu_y for correctness check\n");
            }
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

        printf("Average time for %s: %.9f s\n",               dir->d_name, stats.avg_time_s);
        printf("GFLOP/s for %s: %.6f\n",                      dir->d_name, stats.gflops);
        printf("Standard deviation of time for %s: %.9f s\n", dir->d_name, stats.std_time_s);
        printf("\n");

        perf_stats_write_csv_row(csv, &stats);

        // -------------------------------------------------------
        // CLEANUP (per matrix)

        CHECK_CUSPARSE(cusparseDestroySpMat(matA));
        CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
        CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
        CHECK_CUDA(cudaFree(dBuffer));
        CHECK_CUDA(cudaFree(d_x));
        CHECK_CUDA(cudaFree(d_y));
        CHECK_CUDA(cudaFree(d_csr_A.row_ptr));
        CHECK_CUDA(cudaFree(d_csr_A.col_idx));
        CHECK_CUDA(cudaFree(d_csr_A.values));

        free_coo(&A);
        free_csr(&csr_A);
        free_dense(x);
        free(y);
    }

    closedir(d);
    fclose(csv);
    CHECK_CUSPARSE(cusparseDestroy(handle));
    return 0;
}
