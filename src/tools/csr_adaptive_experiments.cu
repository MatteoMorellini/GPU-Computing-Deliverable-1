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
#define REPS   100
#define WARMUP 5

// -----------------------------------------------------------------------
// Sweep values
// -----------------------------------------------------------------------
static const int NNZ_VALUES[]    = { 256, 512, 1024 };
static const int N_NNZ_VALUES    = (int)(sizeof(NNZ_VALUES)    / sizeof(NNZ_VALUES[0]));

static const int SMALL_VALUES[]  = { 1, 2, 4, 8, 16, 32 };
static const int N_SMALL_VALUES  = (int)(sizeof(SMALL_VALUES)  / sizeof(SMALL_VALUES[0]));

// -----------------------------------------------------------------------
// Kernel — uses dynamic shared memory so the block size is fully runtime.
// Launch as: csr_adaptive_kernel<<<nblocks, nnz_per_block,
//                                  nnz_per_block * sizeof(float)>>>(...);
// -----------------------------------------------------------------------
__global__ void csr_adaptive_kernel(CSR_Matrix mat,
                                    const int * __restrict__ row_blocks,
                                    const float * __restrict__ x,
                                    float * __restrict__ y,
                                    int small_value,
                                    int nnz_per_block) {
    extern __shared__ float lds[];   // size = nnz_per_block * sizeof(float)

    int row_start = row_blocks[blockIdx.x];
    int row_end   = row_blocks[blockIdx.x + 1];
    int num_rows  = row_end - row_start;
    int tid       = threadIdx.x;

    if (num_rows > small_value) {
        // CSR-Stream path
        int first_nz = mat.row_ptr[row_start];
        int last_nz  = mat.row_ptr[row_end];
        int nnz      = last_nz - first_nz;

        if (tid < nnz)
            lds[tid] = mat.values[first_nz + tid] * x[mat.col_idx[first_nz + tid]];
        __syncthreads();

        if (tid < num_rows) {
            int row         = row_start + tid;
            int local_start = mat.row_ptr[row]     - first_nz;
            int local_end   = mat.row_ptr[row + 1] - first_nz;
            float sum = 0.0f;
            for (int j = local_start; j < local_end; j++)
                sum += lds[j];
            y[row] = sum;
        }
    } else {
        // CSR-Vector path
        int row   = row_start;
        int start = mat.row_ptr[row];
        int end   = mat.row_ptr[row + 1];

        float sum = 0.0f;
        for (int j = start + tid; j < end; j += blockDim.x)
            sum += mat.values[j] * x[mat.col_idx[j]];

        lds[tid] = sum;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) lds[tid] += lds[tid + s];
            __syncthreads();
        }

        if (tid == 0) y[row] = lds[0];
    }
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------
void csr_to_device(const CSR_Matrix *h_mat, CSR_Matrix *d_mat) {
    d_mat->rows = h_mat->rows;
    d_mat->cols = h_mat->cols;
    d_mat->nnz  = h_mat->nnz;

    cudaMalloc((void**)&d_mat->row_ptr, (h_mat->rows + 1) * sizeof(int));
    cudaMalloc((void**)&d_mat->col_idx,  h_mat->nnz       * sizeof(int));
    cudaMalloc((void**)&d_mat->values,   h_mat->nnz       * sizeof(float));

    cudaMemcpy(d_mat->row_ptr, h_mat->row_ptr, (h_mat->rows + 1) * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat->col_idx, h_mat->col_idx,  h_mat->nnz       * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat->values,  h_mat->values,   h_mat->nnz       * sizeof(float), cudaMemcpyHostToDevice);
}

// nnz_per_block is now a parameter instead of a macro
int build_row_blocks(const CSR_Matrix *csr, int nnz_per_block, int **row_blocks_out) {
    int *row_blocks = (int*)malloc((size_t)(csr->rows + 1) * sizeof(int));
    if (!row_blocks) { fprintf(stderr, "Error allocating row_blocks\n"); exit(1); }

    int num_blocks     = 0;
    row_blocks[0]      = 0;
    int sum_nnz        = 0;
    int last_block_row = 0;

    for (int row = 0; row < csr->rows; row++) {
        int row_nnz = csr->row_ptr[row + 1] - csr->row_ptr[row];
        sum_nnz += row_nnz;

        if (sum_nnz == nnz_per_block) {
            row_blocks[++num_blocks] = row + 1;
            last_block_row = row + 1;
            sum_nnz = 0;
        } else if (sum_nnz > nnz_per_block) {
            if (row > last_block_row) {
                row_blocks[++num_blocks] = row;
                last_block_row = row;
                row--;
                sum_nnz = 0;
            } else {
                row_blocks[++num_blocks] = row + 1;
                last_block_row = row + 1;
                sum_nnz = 0;
            }
        } else if ((row + 1) - last_block_row == nnz_per_block) {
            row_blocks[++num_blocks] = row + 1;
            last_block_row = row + 1;
            sum_nnz = 0;
        }
    }

    if (row_blocks[num_blocks] != csr->rows)
        row_blocks[++num_blocks] = csr->rows;

    *row_blocks_out = row_blocks;
    return num_blocks;
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------
int main(void) {
    srand(0);

    DIR *d;
    struct dirent *dir;
    char folder[] = "./matrices/";
    char path[1024];

    d = opendir(folder);
    if (!d) { printf("Error opening directory\n"); return 1; }

    printf("Files in matrices directory:\n");

    TIMER_DEF(0);

    while ((dir = readdir(d)) != NULL) {
        if (dir->d_name[0] == '.') continue;
        char *ext = strrchr(dir->d_name, '.');
        if (!ext || strcmp(ext, ".mtx") != 0) continue;

        // -----------------------------------------------------------
        // Load matrix (once per file)
        snprintf(path, sizeof(path), "%s%s", folder, dir->d_name);

        COO_Matrix A;
        read_mtx(path, &A);
        printf("Read matrix %s: %d rows, %d cols, %d non-zeros\n",
               dir->d_name, A.rows, A.cols, A.nnz);

        CSR_Matrix csr_A;
        coo_to_csr(&A, &csr_A);

        // -----------------------------------------------------------
        // Host vectors (once per matrix)
        float *x = (float*)malloc((size_t)csr_A.cols * sizeof(*x));
        float *y = (float*)malloc((size_t)csr_A.rows * sizeof(*y));
        if (!x || !y) {
            fprintf(stderr, "Error allocating host vectors\n");
            free_coo(&A); free_csr(&csr_A); free(x); free(y);
            closedir(d); return 1;
        }
        fill_dense(x, (size_t)csr_A.cols);

        // Upload matrix and x (shared across all sweeps)
        float *d_x = NULL, *d_y = NULL;
        cudaMalloc((void**)&d_x, (size_t)csr_A.cols * sizeof(float));
        cudaMalloc((void**)&d_y, (size_t)csr_A.rows * sizeof(float));
        cudaMemcpy(d_x, x, (size_t)csr_A.cols * sizeof(float), cudaMemcpyHostToDevice);

        CSR_Matrix d_csr_A;
        csr_to_device(&csr_A, &d_csr_A);

        // -----------------------------------------------------------
        // Outer sweep: NNZ_PER_BLOCK
        for (int ni = 0; ni < N_NNZ_VALUES; ni++) {
            int nnz_per_block = NNZ_VALUES[ni];

            // Row blocks depend on nnz_per_block — rebuild for each value
            int *h_row_blocks = NULL;
            int num_blocks = build_row_blocks(&csr_A, nnz_per_block, &h_row_blocks);

            int *d_row_blocks = NULL;
            cudaMalloc((void**)&d_row_blocks, (size_t)(num_blocks + 1) * sizeof(int));
            cudaMemcpy(d_row_blocks, h_row_blocks,
                       (size_t)(num_blocks + 1) * sizeof(int), cudaMemcpyHostToDevice);

            size_t shared_bytes = (size_t)nnz_per_block * sizeof(float);

            printf("  [NNZ_PER_BLOCK=%4d] %d CUDA blocks\n", nnz_per_block, num_blocks);

            // Inner sweep: SMALL_VALUE
            for (int si = 0; si < N_SMALL_VALUES; si++) {
                int small_value = SMALL_VALUES[si];

                double times[REPS];

                // Warmup
                for (int r = 0; r < WARMUP; r++)
                    csr_adaptive_kernel<<<num_blocks, nnz_per_block, shared_bytes>>>(
                        d_csr_A, d_row_blocks, d_x, d_y, small_value, nnz_per_block);
                cudaDeviceSynchronize();

                // Timed runs
                for (int r = 0; r < REPS; r++) {
                    TIMER_START(0);
                    csr_adaptive_kernel<<<num_blocks, nnz_per_block, shared_bytes>>>(
                        d_csr_A, d_row_blocks, d_x, d_y, small_value, nnz_per_block);
                    cudaDeviceSynchronize();
                    TIMER_STOP(0);
                    times[r] = TIMER_ELAPSED(0) / 1e6;
                }

                // Performance metrics
                double total_time = 0.0;
                for (int r = 0; r < REPS; r++) total_time += times[r];
                double avg_time = total_time / REPS;
                double gflops   = (2.0 * csr_A.nnz) / (avg_time * 1e9);

                double variance = 0.0;
                for (int r = 0; r < REPS; r++) {
                    double diff = times[r] - avg_time;
                    variance += diff * diff;
                }
                double std_time = sqrt(variance / REPS);

                printf("    [SMALL_VALUE=%2d] avg=%.9f s | GFlop/s=%.6f | std=%.9f s\n",
                       small_value, avg_time, gflops, std_time);
            }

            cudaFree(d_row_blocks);
            free(h_row_blocks);
            printf("\n");
        }

        // -----------------------------------------------------------
        // Cleanup (per matrix)
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
