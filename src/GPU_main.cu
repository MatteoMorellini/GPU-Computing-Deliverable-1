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
#define VECTOR 1

__global__ void spmv_kernel(CSR_Matrix mat, float *x, float *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < mat.rows) {
        float sum = 0.0f;
        for (int j = mat.row_ptr[row]; j < mat.row_ptr[row + 1]; j++)
            sum += mat.values[j] * x[mat.col_idx[j]];
        y[row] = sum;
    }
}

__global__ void CSR_vector_kernel(const CSR_Matrix mat, const float *x, float *y) {
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    int lane = threadIdx.x % 32;              // thread position inside warp
    int row  = global_thread_id / 32;         // one warp per row

    if (row < mat.rows) {
        int row_start = mat.row_ptr[row];
        int row_end   = mat.row_ptr[row + 1];

        float sum = 0.0f;

        // Each lane processes different nonzeros of the same row
        // The 32 threads in a warp execute this line at the same time (lockstep) so no need for thread sync
        for (int j = row_start + lane; j < row_end; j += 32) {
            sum += mat.values[j] * x[mat.col_idx[j]];
        }

        // Warp-level reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        // Only lane 0 writes the final row result
        if (lane == 0) {
            y[row] = sum;
        }
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

    FILE *csv = perf_stats_open_csv(
        perf_stats_resolve_path(VECTOR ? "results/gpu_csr_vector.csv"
                                       : "results/gpu_csr_scalar.csv"));
    if (!csv) { closedir(d); return 1; }

    TIMER_DEF(0);
    TIMER_DEF(1); // file parse
    TIMER_DEF(2); // format conversion
    TIMER_DEF(3); // H2D transfer

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
        TIMER_START(1);
        read_mtx(path, &A);
        TIMER_STOP(1);
        double file_parse_s = TIMER_ELAPSED(1) / 1e6;
        printf("Read matrix %s: %d rows, %d cols, %d non-zeros\n",
               dir->d_name, A.rows, A.cols, A.nnz);

        CSR_Matrix csr_A;
        TIMER_START(2);
        coo_to_csr(&A, &csr_A);
        TIMER_STOP(2);
        double format_conv_s = TIMER_ELAPSED(2) / 1e6;

        PerfStats stats;
        memset(&stats, 0, sizeof(stats));

        strncpy(stats.name, dir->d_name, sizeof(stats.name) - 1);
        strncpy(stats.format, "CSR", sizeof(stats.format) - 1);
        strncpy(stats.implementation,
                VECTOR ? "CUDA CSR-Vector" : "CUDA CSR-Scalar",
                sizeof(stats.implementation) - 1);

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
        TIMER_START(3);
        csr_to_device(&csr_A, &d_csr_A);
        cudaDeviceSynchronize();
        TIMER_STOP(3);
        double h2d_transfer_s = TIMER_ELAPSED(3) / 1e6;
        // Add this right before cusparseCreateCsr
        printf("row_ptr alignment: %zu\n", (size_t)d_csr_A.row_ptr % 4);
        printf("col_idx alignment: %zu\n", (size_t)d_csr_A.col_idx % 4);
        printf("values  alignment: %zu\n", (size_t)d_csr_A.values  % 4);
        
        // -------------------------------------------------------
        // WARMUP + TIMING SECTION

        double times[REPS];
        if (VECTOR) {
            printf("Running vectorized kernel for %s...\n", dir->d_name);
            int threads_per_block = 256; // 8 warps per block
            int warps_per_block = threads_per_block / 32;
            int blocks = (d_csr_A.rows + warps_per_block - 1) / warps_per_block;
            for (int r = 0; r < WARMUP; r++)
                CSR_vector_kernel<<<blocks, threads_per_block>>>(d_csr_A, d_x, d_y);
                cudaDeviceSynchronize(); 

            for (int r = 0; r < REPS; r++) {
                TIMER_START(0);
                CSR_vector_kernel<<<blocks, threads_per_block>>>(d_csr_A, d_x, d_y);
                cudaDeviceSynchronize(); // forces the CPU to wait until the GPU finishes all previous work.
                TIMER_STOP(0);
                times[r] = TIMER_ELAPSED(0) / 1e6; // microseconds -> seconds
            }
        } else {
            printf("Running scalar kernel for %s...\n", dir->d_name);
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
        }

        

        cudaMemcpy(y, d_y, (size_t)csr_A.rows * sizeof(float), cudaMemcpyDeviceToHost);
        
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
        stats.file_parse_s    = file_parse_s;
        stats.format_conv_s   = format_conv_s;
        stats.h2d_transfer_s  = h2d_transfer_s;


        // -------------------------------------------------------
        printf("Average time for %s: %.9f s\n",            dir->d_name, stats.avg_time_s);
        printf("GFLOP/s for %s: %.6f\n",                   dir->d_name, stats.gflops);
        printf("Standard deviation of time for %s: %.9f s\n", dir->d_name, stats.std_time_s);
        printf("\n");

        perf_stats_write_csv_row(csv, &stats);

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
    fclose(csv);
    return 0;
}