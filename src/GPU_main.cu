#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <stdlib.h>
extern "C" {
    #include "mtx_reader.h"
    #include "coo_to_csr.h"
    #include "csr_spvm.h"
}
#include "gpu_common.cuh"
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
        gpu_init_perf_stats(&stats, dir->d_name, "CSR", "CUDA CSR", &csr_A);

        // -------------------------------------------------------
        // PREPARATION SECTION

        float *x = NULL;
        float *y = NULL;
        if (!gpu_create_host_vectors(&csr_A, &x, &y)) {
            free_coo(&A);
            free_csr(&csr_A);
            closedir(d);
            return 1;
        }

        //-------------------------------------------------------
        // MOVE TO GPU
        float *d_x = NULL;
        float *d_y = NULL;
        gpu_dense_to_device(&csr_A, x, y, &d_x, &d_y);

        CSR_Matrix d_csr_A;
        gpu_csr_to_device(&csr_A, &d_csr_A);
        // Add this right before cusparseCreateCsr
        printf("row_ptr alignment: %zu\n", (size_t)d_csr_A.row_ptr % 4);
        printf("col_idx alignment: %zu\n", (size_t)d_csr_A.col_idx % 4);
        printf("values  alignment: %zu\n", (size_t)d_csr_A.values  % 4);
        
        // -------------------------------------------------------
        // WARMUP + TIMING SECTION

        double times[REPS];
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        if (VECTOR) {
            printf("Running vectorized kernel for %s...\n", dir->d_name);
            int threads_per_block = 256; // 8 warps per block
            int warps_per_block = threads_per_block / 32;
            int blocks = (d_csr_A.rows + warps_per_block - 1) / warps_per_block;
            for (int r = 0; r < WARMUP; r++)
                CSR_vector_kernel<<<blocks, threads_per_block>>>(d_csr_A, d_x, d_y);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());

            for (int r = 0; r < REPS; r++) {
                CHECK_CUDA(cudaEventRecord(start));
                CSR_vector_kernel<<<blocks, threads_per_block>>>(d_csr_A, d_x, d_y);
                CHECK_CUDA(cudaGetLastError());
                CHECK_CUDA(cudaEventRecord(stop));
                CHECK_CUDA(cudaEventSynchronize(stop));

                float elapsed_ms = 0.0f;
                CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
                times[r] = elapsed_ms / 1e3; // milliseconds -> seconds
            }
        } else {
            printf("Running scalar kernel for %s...\n", dir->d_name);
            for (int r = 0; r < WARMUP; r++)
                spmv_kernel<<<(csr_A.rows + 255) / 256, 256>>>(d_csr_A, d_x, d_y);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());

            for (int r = 0; r < REPS; r++) {
                CHECK_CUDA(cudaEventRecord(start));
                spmv_kernel<<<(csr_A.rows + 255) / 256, 256>>>(d_csr_A, d_x, d_y);
                CHECK_CUDA(cudaGetLastError());
                CHECK_CUDA(cudaEventRecord(stop));
                CHECK_CUDA(cudaEventSynchronize(stop));

                float elapsed_ms = 0.0f;
                CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
                times[r] = elapsed_ms / 1e3; // milliseconds -> seconds
            }
        }

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));

        

        gpu_copy_y_to_host(&csr_A, d_y, y);

        // -------------------------------------------------------
        // PERFORMANCE METRICS SECTION

        gpu_finalize_perf_stats(&stats, times, REPS, csr_A.nnz);


        // -------------------------------------------------------
        gpu_print_perf_stats(dir->d_name, &stats);

        gpu_free_dense(d_x, d_y);
        gpu_free_csr(&d_csr_A);

        free_coo(&A);
        free_csr(&csr_A);
        gpu_free_host_vectors(x, y);
    }

    closedir(d);
    return 0;
}
