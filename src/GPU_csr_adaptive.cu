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
        gpu_init_perf_stats(&stats, dir->d_name, "CSR", "CUDA CSR-Adaptive", &csr_A);

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

        int *h_row_blocks = NULL;
        int num_blocks = build_row_blocks(&csr_A, &h_row_blocks);
        printf("  CSR-Adaptive: %d CUDA blocks (target %d nnz/block)\n",
               num_blocks, NNZ_PER_BLOCK);

        //-------------------------------------------------------
        // MOVE TO GPU
        float *d_x = NULL;
        float *d_y = NULL;
        gpu_dense_to_device(&csr_A, x, y, &d_x, &d_y);

        CSR_Matrix d_csr_A;
        gpu_csr_to_device(&csr_A, &d_csr_A);

        int *d_row_blocks = NULL;
        CHECK_CUDA(cudaMalloc((void**)&d_row_blocks, (size_t)(num_blocks + 1) * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(d_row_blocks, h_row_blocks,
                              (size_t)(num_blocks + 1) * sizeof(int),
                              cudaMemcpyHostToDevice));

        // -------------------------------------------------------
        // WARMUP + TIMING SECTION

        double times[REPS];
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        for (int r = 0; r < WARMUP; r++)
            csr_adaptive_kernel<<<num_blocks, NNZ_PER_BLOCK>>>(d_csr_A, d_row_blocks, d_x, d_y);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        for (int r = 0; r < REPS; r++) {
            CHECK_CUDA(cudaEventRecord(start));
            csr_adaptive_kernel<<<num_blocks, NNZ_PER_BLOCK>>>(d_csr_A, d_row_blocks, d_x, d_y);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));

            float elapsed_ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
            times[r] = elapsed_ms / 1e3; // milliseconds -> seconds
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
        CHECK_CUDA(cudaFree(d_row_blocks));
        gpu_free_csr(&d_csr_A);

        free(h_row_blocks);
        free_coo(&A);
        free_csr(&csr_A);
        gpu_free_host_vectors(x, y);
    }

    closedir(d);
    return 0;
}
