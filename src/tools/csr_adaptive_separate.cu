#include "gpu_bench.cuh"
#include <vector>

#define NNZ_PER_WG     1024
#define THREADS_PER_WG 256
#define SMALL_VALUE    2

// =============================================================================
// CSR-Stream kernel (paper Algorithm 3): row-block has > SMALL_VALUE rows
// whose combined nnz fits in NNZ_PER_WG. Stage products into LDS, then one
// thread per row scalar-sums its slice.
// =============================================================================
__global__ void csr_stream_paper_kernel(CSR_Matrix mat,
                                        const int * __restrict__ block_row_start,
                                        const int * __restrict__ block_row_end,
                                        const float * __restrict__ x,
                                        float * __restrict__ y) {
    __shared__ float LDS[NNZ_PER_WG];

    const int startRow     = block_row_start[blockIdx.x];
    const int nextStartRow = block_row_end[blockIdx.x];
    const int num_rows     = nextStartRow - startRow;
    const int localTID     = threadIdx.x;

    const int first_col      = mat.row_ptr[startRow];
    const int num_non_zeroes = mat.row_ptr[nextStartRow] - first_col;

    // Stream num_non_zeroes values into LDS with coalesced, strided loads.
    for (int i = localTID; i < num_non_zeroes; i += THREADS_PER_WG) {
        LDS[i] = mat.values[first_col + i] * x[mat.col_idx[first_col + i]];
    }
    __syncthreads();

    // Scalar-style reduction: one thread per row. If num_rows exceeds
    // THREADS_PER_WG, sweep with a while loop.
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
}

// =============================================================================
// CSR-Vector kernel (paper Algorithm 3 fallback): row-block has <= SMALL_VALUE
// rows. One warp per row, reads directly from global memory.
// =============================================================================
__global__ void csr_vector_paper_kernel(CSR_Matrix mat,
                                        const int * __restrict__ block_row_start,
                                        const int * __restrict__ block_row_end,
                                        const float * __restrict__ x,
                                        float * __restrict__ y) {
    const int startRow     = block_row_start[blockIdx.x];
    const int nextStartRow = block_row_end[blockIdx.x];
    const int num_rows     = nextStartRow - startRow;
    const int localTID     = threadIdx.x;
    const int warp_id      = localTID >> 5;
    const int lane         = localTID & 31;

    if (warp_id < num_rows) {
        const int row   = startRow + warp_id;
        const int start = mat.row_ptr[row];
        const int end   = mat.row_ptr[row + 1];

        float sum = 0.0f;
        for (int j = start + lane; j < end; j += 32) {
            sum += mat.values[j] * x[mat.col_idx[j]];
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) y[row] = sum;
    }
}

// =============================================================================
// Host-side block builder (Algorithm 2 of the paper) — split into Stream and
// Vector groups so each can be launched as its own kernel.
// =============================================================================
struct PaperBlockPlan {
    std::vector<int> sstart, send;   // Stream blocks  (num_rows >  SMALL_VALUE)
    std::vector<int> vstart, vend;   // Vector blocks  (num_rows <= SMALL_VALUE)
};

static void emit(PaperBlockPlan &plan, int s, int e) {
    if (e - s > SMALL_VALUE) {
        plan.sstart.push_back(s);
        plan.send.push_back(e);
    } else {
        plan.vstart.push_back(s);
        plan.vend.push_back(e);
    }
}

static void build_paper_plan(const CSR_Matrix *csr, PaperBlockPlan &plan) {
    const int totalRows = csr->rows;
    const int *row_delimiters = csr->row_ptr;

    int prev   = 0;
    int sum    = 0;
    int last_i = 0;

    for (int i = 1; i <= totalRows; i++) {
        sum += row_delimiters[i] - row_delimiters[i - 1];

        if (sum == NNZ_PER_WG) {
            emit(plan, prev, i);
            prev   = i;
            last_i = i;
            sum    = 0;
        } else if (sum > NNZ_PER_WG) {
            if (i - last_i > 1) {
                // The extra row will not fit: close block before row i, retry.
                emit(plan, prev, i - 1);
                prev   = i - 1;
                last_i = i - 1;
                sum    = 0;
                i--;
            } else { // i - last_i == 1: this single row is too large.
                emit(plan, prev, i);
                prev   = i;
                last_i = i;
                sum    = 0;
            }
        }
    }

    if (prev != totalRows) {
        emit(plan, prev, totalRows);
    }
}

struct CsrAdaptivePaperImpl {
    int *d_sstart = nullptr, *d_send = nullptr;   // Stream blocks
    int *d_vstart = nullptr, *d_vend = nullptr;   // Vector blocks
    int num_stream = 0, num_vector = 0;

    static void upload_pair(std::vector<int> &a, std::vector<int> &b,
                            int **d_a, int **d_b) {
        const size_t n = a.size();
        if (n == 0) return;
        cudaMalloc((void**)d_a, n * sizeof(int));
        cudaMalloc((void**)d_b, n * sizeof(int));
        cudaMemcpy(*d_a, a.data(), n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(*d_b, b.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    }

    double prep(const CSR_Matrix &h_csr, const CSR_Matrix &,
                float *, float *) {
        TIMER_DEF(t);
        TIMER_START(t);

        PaperBlockPlan plan;
        build_paper_plan(&h_csr, plan);
        num_stream = (int)plan.sstart.size();
        num_vector = (int)plan.vstart.size();

        upload_pair(plan.sstart, plan.send, &d_sstart, &d_send);
        upload_pair(plan.vstart, plan.vend, &d_vstart, &d_vend);

        cudaDeviceSynchronize();
        TIMER_STOP(t);
        printf("  CSR-Adaptive (paper): %d stream + %d vector blocks, "
               "%d threads/block, %d nnz/block, SMALL_VALUE=%d\n",
               num_stream, num_vector,
               THREADS_PER_WG, NNZ_PER_WG, SMALL_VALUE);
        return TIMER_ELAPSED(t) / 1e6;
    }

    void launch(const CSR_Matrix &d_csr, const float *d_x, float *d_y) {
        // Two independent launches -> Nsight reports timing per path.
        if (num_stream > 0)
            csr_stream_paper_kernel<<<num_stream, THREADS_PER_WG>>>(
                d_csr, d_sstart, d_send, d_x, d_y);
        if (num_vector > 0)
            csr_vector_paper_kernel<<<num_vector, THREADS_PER_WG>>>(
                d_csr, d_vstart, d_vend, d_x, d_y);
    }

    void teardown() {
        if (d_sstart) cudaFree(d_sstart);
        if (d_send)   cudaFree(d_send);
        if (d_vstart) cudaFree(d_vstart);
        if (d_vend)   cudaFree(d_vend);
        d_sstart = d_send = d_vstart = d_vend = nullptr;
        num_stream = num_vector = 0;
    }
};

int main(void) {
    srand(0);
    CsrAdaptivePaperImpl impl;
    BenchConfig cfg{"CUDA CSR-Adaptive (paper)", "CSR",
                    "results/csr_adaptive_paper.csv"};
    return run_benchmark(cfg, impl);
}
