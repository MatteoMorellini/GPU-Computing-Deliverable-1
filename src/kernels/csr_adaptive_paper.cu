#include "gpu_bench.cuh"

#define NNZ_PER_WG     1024
#define THREADS_PER_WG 256
#define SMALL_VALUE    2

__global__ void csr_adaptive_paper_kernel(CSR_Matrix mat,
                                          const int * __restrict__ row_blocks,
                                          const float * __restrict__ x,
                                          float * __restrict__ y) {
    __shared__ float LDS[NNZ_PER_WG];

    const int startRow     = row_blocks[blockIdx.x];
    const int nextStartRow = row_blocks[blockIdx.x + 1];
    const int num_rows     = nextStartRow - startRow;
    const int localTID     = threadIdx.x;

    if (num_rows > SMALL_VALUE) {
        const int first_col      = mat.row_ptr[startRow];
        const int num_non_zeroes = mat.row_ptr[nextStartRow] - first_col;

        // Stream num_non_zeroes values into LDS with coalesced, strided loads.
        for (int i = localTID; i < num_non_zeroes; i += THREADS_PER_WG) {
            LDS[i] = mat.values[first_col + i] * x[mat.col_idx[first_col + i]];
        }
        __syncthreads();

        // Scalar-style reduction: one thread per row. If num_rows exceeds
        // THREADS_PER_WG, sweep with a while loop
        int rows_done = 0;
        while (rows_done < num_rows) {
            const int local_row = rows_done + localTID;
            if (local_row < num_rows) {
                const int row     = startRow + local_row;
                const int t_start = mat.row_ptr[row]     - first_col;
                const int t_end   = mat.row_ptr[row + 1] - first_col;
                float temp = 0.0f;
                for (int j = t_start; j < t_end; j++) {
                    temp += LDS[j];
                }
                y[row] = temp;
            }
            rows_done += THREADS_PER_WG;
        }
    } else {
        // With 256 threads we have 8 warps available; we assign warp `warp_id` to the
        // row at offset `warp_id` (only warps 0 and 1 do useful work).
        const int warp_id = localTID >> 5;
        const int lane    = localTID & 31;

        if (warp_id < num_rows) {
            const int row   = startRow + warp_id;
            const int start = mat.row_ptr[row];
            const int end   = mat.row_ptr[row + 1];

            float sum = 0.0f;
            for (int j = start + lane; j < end; j += 32) {
                sum += mat.values[j] * x[mat.col_idx[j]];
            }
            // Logarithmic warp reduction (paper: log-reduce at end of Vector).
            for (int offset = 16; offset > 0; offset >>= 1) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
            if (lane == 0) y[row] = sum;
        }
    }
}

static int build_row_blocks_paper(const CSR_Matrix *csr, int **row_blocks_out) {
    const int totalRows = csr->rows;
    const int *row_delimiters = csr->row_ptr;

    int *row_blocks = (int*)malloc((size_t)(totalRows + 2) * sizeof(int));
    if (!row_blocks) {
        fprintf(stderr, "Error allocating row_blocks\n");
        exit(1);
    }

    row_blocks[0] = 0;
    int sum    = 0;
    int last_i = 0;
    int ctr    = 1;

    for (int i = 1; i <= totalRows; i++) {
        sum += row_delimiters[i] - row_delimiters[i - 1];

        if (sum == NNZ_PER_WG) {
            last_i = i;
            row_blocks[ctr++] = i;
            sum = 0;
        } else if (sum > NNZ_PER_WG) {
            if (i - last_i > 1) {
                // The extra row will not fit: close block before row i, retry.
                row_blocks[ctr++] = i - 1;
                i--;
            } else { // i - last_i == 1: this single row is too large.
                row_blocks[ctr++] = i;
            }
            last_i = i;
            sum = 0;
        }
    }

    if (row_blocks[ctr - 1] != totalRows) {
        row_blocks[ctr++] = totalRows;
    }

    *row_blocks_out = row_blocks;
    return ctr - 1; // number of blocks
}

struct CsrAdaptivePaperImpl {
    int *d_row_blocks = nullptr;
    int  num_blocks   = 0;

    double prep(const CSR_Matrix &h_csr, const CSR_Matrix &,
                float *, float *) {
        TIMER_DEF(t);
        TIMER_START(t);
        int *h_row_blocks = nullptr;
        num_blocks = build_row_blocks_paper(&h_csr, &h_row_blocks);
        cudaMalloc((void**)&d_row_blocks,
                   (size_t)(num_blocks + 1) * sizeof(int));
        cudaMemcpy(d_row_blocks, h_row_blocks,
                   (size_t)(num_blocks + 1) * sizeof(int),
                   cudaMemcpyHostToDevice);
        free(h_row_blocks);
        cudaDeviceSynchronize();
        TIMER_STOP(t);
        printf("  CSR-Adaptive (paper): %d blocks, %d threads/block, "
               "%d nnz/block, SMALL_VALUE=%d\n",
               num_blocks, THREADS_PER_WG, NNZ_PER_WG, SMALL_VALUE);
        return TIMER_ELAPSED(t) / 1e6;
    }

    void launch(const CSR_Matrix &d_csr, const float *d_x, float *d_y) {
        csr_adaptive_paper_kernel<<<num_blocks, THREADS_PER_WG>>>(
            d_csr, d_row_blocks, d_x, d_y);
    }

    void teardown() {
        if (d_row_blocks) {
            cudaFree(d_row_blocks);
            d_row_blocks = nullptr;
        }
        num_blocks = 0;
    }
};

int main(void) {
    srand(0);
    CsrAdaptivePaperImpl impl;
    BenchConfig cfg{"CUDA CSR-Adaptive (paper)", "CSR",
                    "results/csr_adaptive_paper.csv"};
    return run_benchmark(cfg, impl);
}
