#include "gpu_bench.cuh"

#define NNZ_PER_BLOCK   256
#define STREAM_MIN_ROWS 8

// -----------------------------------------------------------------------------
// CSR-Adaptive SpMV kernel (Greathouse & Daga).
// Each CUDA block is assigned a contiguous range of rows [row_blocks[bid],
// row_blocks[bid+1]). The preprocessing guarantees that either:
//   (a) the range covers many rows (> STREAM_MIN_ROWS) whose total nnz fits in
//       NNZ_PER_BLOCK
//       -> CSR-Stream: every thread loads one nonzero product into shared
//          memory, then each row is reduced sequentially by one thread.
//   (b) the range covers few rows (1..STREAM_MIN_ROWS), potentially long
//       -> CSR-Vector: each warp of the block handles one of the rows.
// -----------------------------------------------------------------------------
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
        int first_nz = mat.row_ptr[row_start];
        int last_nz  = mat.row_ptr[row_end];
        int nnz      = last_nz - first_nz;

        if (tid < nnz) {
            lds[tid] = mat.values[first_nz + tid]
                     * x[mat.col_idx[first_nz + tid]];
        }
        __syncthreads();

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
        int warp_id = tid >> 5;
        int lane    = tid & 31;

        if (warp_id < num_rows) {
            int row   = row_start + warp_id;
            int start = mat.row_ptr[row];
            int end   = mat.row_ptr[row + 1];

            float sum = 0.0f;
            for (int j = start + lane; j < end; j += 32)
                sum += mat.values[j] * x[mat.col_idx[j]];

            for (int offset = 16; offset > 0; offset >>= 1)
                sum += __shfl_down_sync(0xffffffff, sum, offset);

            if (lane == 0) y[row] = sum;
        }
    }
}

// -----------------------------------------------------------------------------
// Build the row_blocks array: row boundaries such that every block either
// packs consecutive short rows with total nnz <= NNZ_PER_BLOCK, or owns one
// long row exclusively. Returns the number of CUDA blocks.
// -----------------------------------------------------------------------------
static int build_row_blocks(const CSR_Matrix *csr, int **row_blocks_out) {
    int *row_blocks = (int*)malloc((size_t)(csr->rows + 1) * sizeof(int));
    if (!row_blocks) {
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
                row_blocks[++num_blocks] = row;
                last_block_row = row;
                row--;
                sum_nnz = 0;
            } else {
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

struct CsrAdaptiveImpl {
    int *d_row_blocks = nullptr;
    int  num_blocks   = 0;

    double prep(const CSR_Matrix &h_csr, const CSR_Matrix &,
                float *, float *) {
        TIMER_DEF(t);
        TIMER_START(t);
        int *h_row_blocks = nullptr;
        num_blocks = build_row_blocks(&h_csr, &h_row_blocks);
        cudaMalloc((void**)&d_row_blocks,
                   (size_t)(num_blocks + 1) * sizeof(int));
        cudaMemcpy(d_row_blocks, h_row_blocks,
                   (size_t)(num_blocks + 1) * sizeof(int),
                   cudaMemcpyHostToDevice);
        free(h_row_blocks);
        cudaDeviceSynchronize();
        TIMER_STOP(t);
        printf("  CSR-Adaptive: %d CUDA blocks (target %d nnz/block)\n",
               num_blocks, NNZ_PER_BLOCK);
        return TIMER_ELAPSED(t) / 1e6;
    }

    void launch(const CSR_Matrix &d_csr, const float *d_x, float *d_y) {
        csr_adaptive_kernel<<<num_blocks, NNZ_PER_BLOCK>>>(
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
    CsrAdaptiveImpl impl;
    BenchConfig cfg{"CUDA CSR-Adaptive", "CSR", "results/csr_adaptive.csv"};
    return run_benchmark(cfg, impl);
}
