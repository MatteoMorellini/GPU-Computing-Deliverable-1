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
#define REPS 100
#define WARMUP 5

__global__ void spmv_kernel(CSR_Matrix mat, float *x, float *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < mat.rows) {
        float sum = 0.0f;
        for (int j = mat.row_ptr[row]; j < mat.row_ptr[row + 1]; j++)
            sum += mat.values[j] * x[mat.col_idx[j]];
        y[row] = sum;
    }
}

// Allocate and transfer CSR matrix from host to device
void csr_to_device(const CSR_Matrix *h_mat, CSR_Matrix *d_mat) {
    // Copy scalar fields directly
    d_mat->rows = h_mat->rows;
    d_mat->cols = h_mat->cols;
    d_mat->nnz  = h_mat->nnz;

    // Allocate device memory for arrays
    cudaMalloc((void**)&d_mat->row_ptr, (h_mat->rows + 1) * sizeof(int));
    cudaMalloc((void**)&d_mat->col_idx,  h_mat->nnz       * sizeof(int));
    cudaMalloc((void**)&d_mat->values,   h_mat->nnz       * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_mat->row_ptr, h_mat->row_ptr,
               (h_mat->rows + 1) * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat->col_idx, h_mat->col_idx,
               h_mat->nnz        * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat->values,  h_mat->values,
               h_mat->nnz        * sizeof(float), cudaMemcpyHostToDevice);
}

int main(void) {
    srand(0); // set seed for reproducibility

    DIR *d;
    struct dirent *dir;
    char folder[] = "./matrices/";
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

        float *x = (float*)malloc((size_t)csr_A.cols * sizeof(*x));

        if (x == NULL) {
            fprintf(stderr, "Error allocating memory for dense vector\n");
            free_coo(&A);
            free_csr(&csr_A);
            closedir(d);
            return 1;
        }
        fill_dense(x, (size_t)csr_A.cols);

        float *y = (float*)malloc((size_t)csr_A.rows * sizeof(*y));
        if (y == NULL) {
            fprintf(stderr, "Error: could not allocate output vector y\n");
            free_coo(&A);
            free_csr(&csr_A);
            free_dense(x);
            closedir(d);
            return 1;
        }

        //-------------------------------------------------------
        // MOVE TO GPU
        float *d_x = NULL;
        float *d_y = NULL;
        cudaMalloc((void**)&d_x, (size_t)csr_A.cols * sizeof(float));
        cudaMalloc((void**)&d_y, (size_t)csr_A.rows * sizeof(float));
        cudaMemcpy(d_x, x, (size_t)csr_A.cols * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, (size_t)csr_A.rows * sizeof(float), cudaMemcpyHostToDevice);

        /* cudaError_t cudaMalloc(void **devPtr, size_t size); needs a pointer-to-pointer to store the device address, 
           instead of returning it directly like malloc: void *malloc(size_t size);
        
        */
        CSR_Matrix d_csr_A;
        csr_to_device(&csr_A, &d_csr_A); 
        
        // -------------------------------------------------------
        // WARMUP + TIMING SECTION

        double times[REPS];

        for (int r = 0; r < WARMUP; r++)
            spmv_kernel<<<(csr_A.rows + 255) / 256, 256>>>(d_csr_A, d_x, d_y);
            cudaDeviceSynchronize(); 

        for (int r = 0; r < REPS; r++) {
            TIMER_START(0);
            spmv_kernel<<<(csr_A.rows + 255) / 256, 256>>>(d_csr_A, d_x, d_y);
            cudaDeviceSynchronize(); // forces the CPU to wait until the GPU finishes all previous work.
            TIMER_STOP(0);
            times[r] = TIMER_ELAPSED(0) / 1e6; // microseconds -> seconds
        }

        cudaMemcpy(y, d_y, (size_t)csr_A.rows * sizeof(float), cudaMemcpyDeviceToHost);

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
        printf("Average time for %s: %.9f s\n",            dir->d_name, stats.avg_time_s);
        printf("GFLOP/s for %s: %.6f\n",                   dir->d_name, stats.gflops);
        printf("Standard deviation of time for %s: %.9f s\n", dir->d_name, stats.std_time_s);
        printf("\n");

        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_csr_A.row_ptr);
        cudaFree(d_csr_A.col_idx);
        cudaFree(d_csr_A.values);

        free_coo(&A);
        free_csr(&csr_A);
        free_dense(x);
        free(y);
    }

    closedir(d);
    return 0;
}