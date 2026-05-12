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

// Threads per CUDA block. Also the target nonzero workload per block during
// preprocessing: stream rows are packed up to this many nonzeros, and vector
// rows reduce across this many threads.
#define NNZ_PER_BLOCK 256

// -----------------------------------------------------------------------------
// CSR-Adaptive SpMV kernel (Greathouse & Daga).
// Each CUDA block is assigned a contiguous range of rows [row_blocks[bid],
// row_blocks[bid+1]). The preprocessing guarantees that either:
//   (a) the range covers many rows (> STREAM_MIN_ROWS) whose total nnz fits in
//       NNZ_PER_BLOCK
//       -> CSR-Stream: every thread loads one nonzero product into shared
//          memory, then each row is reduced sequentially by one thread.
//   (b) the range covers few rows (1..STREAM_MIN_ROWS), potentially long
//       -> CSR-Vector: the whole CUDA block strides through each row in turn
//          and performs a shared-memory tree reduction per row.
// -----------------------------------------------------------------------------
#define STREAM_MIN_ROWS 8

__global__ void csr_adaptive_kernel(CSR_Matrix mat,
                                    const int * __restrict__ row_blocks,
                                    const float * __restrict__ x,
                                    float * __restrict__ y) {
    __shared__ float lds[NNZ_PER_BLOCK];

    int row_start = row_blocks[blockIdx.x];
    int row_end   = row_blocks[blockIdx.x + 1];
    int num_rows  = row_end - row_start;
    int tid       = threadIdx.x;

    if (num_rows > STREAM_MIN_ROWS) {
        // CSR-Stream: all nonzeros of this row range fit in NNZ_PER_BLOCK.
        int first_nz = mat.row_ptr[row_start];
        int last_nz  = mat.row_ptr[row_end];
        int nnz      = last_nz - first_nz;

        if (tid < nnz) {
            lds[tid] = mat.values[first_nz + tid]
                     * x[mat.col_idx[first_nz + tid]];
        }
        __syncthreads();

        // One thread per row reduces its slice of lds.
        if (tid < num_rows) {
            int row = row_start + tid;
            int local_start = mat.row_ptr[row]     - first_nz;
            int local_end   = mat.row_ptr[row + 1] - first_nz;
            float sum = 0.0f;
            for (int j = local_start; j < local_end; j++)
                sum += lds[j];
            y[row] = sum;
        }
    } else {
        // CSR-Vector (warp-per-row): each warp handles one row of the block.
        // Requires blockDim.x to be a multiple of 32; with NNZ_PER_BLOCK = 256
        // we have 8 warps, matching the STREAM_MIN_ROWS cap.
        // the implicit assumption is: STREAM_MIN_ROWS ≤ blockDim.x / 32
        int warp_id = tid >> 5;
        int lane    = tid & 31;

        if (warp_id < num_rows) {
            int row   = row_start + warp_id;
            int start = mat.row_ptr[row];
            int end   = mat.row_ptr[row + 1];

            float sum = 0.0f;
            for (int j = start + lane; j < end; j += 32)
                sum += mat.values[j] * x[mat.col_idx[j]];

            // Warp-level tree reduction via shuffles (no shared memory).
            for (int offset = 16; offset > 0; offset >>= 1)
                sum += __shfl_down_sync(0xffffffff, sum, offset);

            if (lane == 0) y[row] = sum;
        }
    }
}

// Allocate and transfer CSR matrix from host to device
void csr_to_device(const CSR_Matrix *h_mat, CSR_Matrix *d_mat) {
    d_mat->rows = h_mat->rows;
    d_mat->cols = h_mat->cols;
    d_mat->nnz  = h_mat->nnz;

    cudaMalloc((void**)&d_mat->row_ptr, (h_mat->rows + 1) * sizeof(int));
    cudaMalloc((void**)&d_mat->col_idx,  h_mat->nnz       * sizeof(int));
    cudaMalloc((void**)&d_mat->values,   h_mat->nnz       * sizeof(float));

    cudaMemcpy(d_mat->row_ptr, h_mat->row_ptr,
               (h_mat->rows + 1) * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat->col_idx, h_mat->col_idx,
               h_mat->nnz        * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat->values,  h_mat->values,
               h_mat->nnz        * sizeof(float), cudaMemcpyHostToDevice);
}

// -----------------------------------------------------------------------------
// Build the row_blocks array: a list of row boundaries such that every block
// either packs consecutive short rows with total nnz <= NNZ_PER_BLOCK, or
// owns a single long row exclusively. Returns the number of CUDA blocks.
// -----------------------------------------------------------------------------
int build_row_blocks(const CSR_Matrix *csr, int **row_blocks_out) {
    // Upper bound: at most rows + 1 entries (one block per row in the worst case).
    int *row_blocks = (int*)malloc((size_t)(csr->rows + 1) * sizeof(int));
    if (row_blocks == NULL) {
        fprintf(stderr, "Error allocating row_blocks\n");
        exit(1);
    }

    int num_blocks = 0;
    row_blocks[0] = 0;
    int sum_nnz = 0;
    int last_block_row = 0;

    for (int row = 0; row < csr->rows; row++) {
        int row_nnz = csr->row_ptr[row + 1] - csr->row_ptr[row];
        sum_nnz += row_nnz;

        if (sum_nnz == NNZ_PER_BLOCK) {
            row_blocks[++num_blocks] = row + 1;
            last_block_row = row + 1;
            sum_nnz = 0;
        } else if (sum_nnz > NNZ_PER_BLOCK) {
            if (row > last_block_row) {
                // Back off: close the previous block before this row, then
                // restart accumulation with this row.
                row_blocks[++num_blocks] = row;
                last_block_row = row;
                row--;          // reprocess current row in the next block
                sum_nnz = 0;
            } else {
                // A single row exceeds NNZ_PER_BLOCK -> dedicate a block to it.
                row_blocks[++num_blocks] = row + 1;
                last_block_row = row + 1;
                sum_nnz = 0;
            }
        }
    }

    if (row_blocks[num_blocks] != csr->rows) {
        row_blocks[++num_blocks] = csr->rows;
    }

    *row_blocks_out = row_blocks;
    return num_blocks;
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
        strncpy(stats.implementation, "CUDA CSR-Adaptive", sizeof(stats.implementation) - 1);

        stats.rows  = csr_A.rows;
        stats.cols  = csr_A.cols;
        stats.nnz   = csr_A.nnz;
        stats.valid = 1;

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

        int *h_row_blocks = NULL;
        int num_blocks = build_row_blocks(&csr_A, &h_row_blocks);
        printf("  CSR-Adaptive: %d CUDA blocks (target %d nnz/block)\n",
               num_blocks, NNZ_PER_BLOCK);

        //-------------------------------------------------------
        // MOVE TO GPU
        float *d_x = NULL;
        float *d_y = NULL;
        cudaMalloc((void**)&d_x, (size_t)csr_A.cols * sizeof(float));
        cudaMalloc((void**)&d_y, (size_t)csr_A.rows * sizeof(float));
        cudaMemcpy(d_x, x, (size_t)csr_A.cols * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, (size_t)csr_A.rows * sizeof(float), cudaMemcpyHostToDevice);

        CSR_Matrix d_csr_A;
        csr_to_device(&csr_A, &d_csr_A);

        int *d_row_blocks = NULL;
        cudaMalloc((void**)&d_row_blocks, (size_t)(num_blocks + 1) * sizeof(int));
        cudaMemcpy(d_row_blocks, h_row_blocks,
                   (size_t)(num_blocks + 1) * sizeof(int),
                   cudaMemcpyHostToDevice);

        // -------------------------------------------------------
        // WARMUP + TIMING SECTION

        double times[REPS];

        for (int r = 0; r < WARMUP; r++)
            csr_adaptive_kernel<<<num_blocks, NNZ_PER_BLOCK>>>(d_csr_A, d_row_blocks, d_x, d_y);
        cudaDeviceSynchronize();

        for (int r = 0; r < REPS; r++) {
            TIMER_START(0);
            csr_adaptive_kernel<<<num_blocks, NNZ_PER_BLOCK>>>(d_csr_A, d_row_blocks, d_x, d_y);
            cudaDeviceSynchronize();
            TIMER_STOP(0);
            times[r] = TIMER_ELAPSED(0) / 1e6; // microseconds -> seconds
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

        // -------------------------------------------------------
        printf("Average time for %s: %.9f s\n",            dir->d_name, stats.avg_time_s);
        printf("GFLOP/s for %s: %.6f\n",                   dir->d_name, stats.gflops);
        printf("Standard deviation of time for %s: %.9f s\n", dir->d_name, stats.std_time_s);
        printf("\n");

        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_row_blocks);
        cudaFree(d_csr_A.row_ptr);
        cudaFree(d_csr_A.col_idx);
        cudaFree(d_csr_A.values);

        free(h_row_blocks);
        free_coo(&A);
        free_csr(&csr_A);
        free_dense(x);
        free(y);
    }

    closedir(d);
    return 0;
}
