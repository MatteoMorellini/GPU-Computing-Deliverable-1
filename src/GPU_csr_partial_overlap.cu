#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include <cuda/pipeline>
#include <cuda/barrier>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

extern "C" {
    #include "mtx_reader.h"
    #include "coo_to_csr.h"
    #include "generate_dense.h"
    #include "csr_spvm.h"
    #include "time_lib.h"
    #include "perf_stats.h"
}

namespace cg = cooperative_groups;

#define REPS    100
#define WARMUP  5

// Paper: max_batch_size >= 1024 gives well-balanced batches and rare long rows.
#define MAX_BATCH_SIZE  1024
#define THREADS_PER_BLOCK 128

// Switch points for vector_size selection vs. mean row length (Table 2).
#define SWITCH_POINT_1   17
#define SWITCH_POINT_2   34
#define SWITCH_POINT_4   64
#define SWITCH_POINT_8  122
#define SWITCH_POINT_16 216

typedef struct {
    int start_row;   // inclusive
    int end_row;     // exclusive
    int start_nnz;
    int end_nnz;     // exclusive
} BatchInfo;

// =============================================================================
// Host-side: dynamic batch partition (Algorithm 3).
//   - Extra-long rows (nnz > MAX_BATCH_SIZE) are recorded in long_rows.
//   - The remaining rows are packed greedily into batches, each batch holding
//     a whole-number of rows whose nnz sum is <= MAX_BATCH_SIZE. This avoids
//     splitting any row across batches (so no partial sums to fix up).
// =============================================================================
static int build_batches(const CSR_Matrix *csr,
                         BatchInfo **batches_out,
                         int **long_rows_out,
                         int *long_row_count_out) {
    int m = csr->rows;

    int *long_rows = (int*)malloc((size_t)m * sizeof(int));
    BatchInfo *batches = (BatchInfo*)malloc((size_t)(m + 1) * sizeof(BatchInfo));
    if (!long_rows || !batches) {
        fprintf(stderr, "Error allocating batch arrays\n");
        exit(1);
    }

    int long_count = 0;
    int batch_count = 0;

    int cur_start_row = -1;
    int cur_nnz_sum = 0;

    for (int row = 0; row < m; row++) {
        int row_nnz = csr->row_ptr[row + 1] - csr->row_ptr[row];

        if (row_nnz > MAX_BATCH_SIZE) {
            // Flush any pending batch, then record the long row.
            if (cur_start_row >= 0) {
                batches[batch_count].start_row = cur_start_row;
                batches[batch_count].end_row   = row;
                batches[batch_count].start_nnz = csr->row_ptr[cur_start_row];
                batches[batch_count].end_nnz   = csr->row_ptr[row];
                batch_count++;
                cur_start_row = -1;
                cur_nnz_sum = 0;
            }
            long_rows[long_count++] = row;
            continue;
        }

        if (cur_start_row < 0) {
            cur_start_row = row;
            cur_nnz_sum = row_nnz;
        } else if (cur_nnz_sum + row_nnz > MAX_BATCH_SIZE) {
            // Close the previous batch, start a new one with `row`.
            batches[batch_count].start_row = cur_start_row;
            batches[batch_count].end_row   = row;
            batches[batch_count].start_nnz = csr->row_ptr[cur_start_row];
            batches[batch_count].end_nnz   = csr->row_ptr[row];
            batch_count++;
            cur_start_row = row;
            cur_nnz_sum = row_nnz;
        } else {
            cur_nnz_sum += row_nnz;
        }
    }

    if (cur_start_row >= 0) {
        batches[batch_count].start_row = cur_start_row;
        batches[batch_count].end_row   = m;
        batches[batch_count].start_nnz = csr->row_ptr[cur_start_row];
        batches[batch_count].end_nnz   = csr->row_ptr[m];
        batch_count++;
    }

    *batches_out = batches;
    *long_rows_out = long_rows;
    *long_row_count_out = long_count;
    return batch_count;
}

// =============================================================================
// Device-side compute (Algorithm 5): a "vector" of VECTOR_SIZE threads reduces
// one row at a time; the block has THREADS_PER_BLOCK / VECTOR_SIZE vectors that
// stride over the rows of the batch.
// =============================================================================
template <int VECTOR_SIZE>
__device__ inline void vector_compute(const BatchInfo batch,
                                      const int * __restrict__ row_ptr,
                                      const int * __restrict__ scol,
                                      const float * __restrict__ sval,
                                      const float * __restrict__ x,
                                      float * __restrict__ y) {
    int tid = threadIdx.x;
    int vector_id   = tid / VECTOR_SIZE;
    int vector_num  = blockDim.x / VECTOR_SIZE;
    int lane_id     = tid & (VECTOR_SIZE - 1);

    int batch_start_nnz = batch.start_nnz;

    for (int row = batch.start_row + vector_id;
         row < batch.end_row;
         row += vector_num) {

        int nz_start = row_ptr[row];
        int nz_end   = row_ptr[row + 1];

        float sum = 0.0f;
        for (int j = nz_start + lane_id; j < nz_end; j += VECTOR_SIZE) {
            int local = j - batch_start_nnz;
            sum += sval[local] * x[scol[local]];
        }

        // Intra-vector reduction (warp shuffles within a sub-warp of VECTOR_SIZE).
        if (VECTOR_SIZE > 1) {
            unsigned mask = __activemask();
            #pragma unroll
            for (int off = VECTOR_SIZE >> 1; off > 0; off >>= 1) {
                sum += __shfl_down_sync(mask, sum, off, VECTOR_SIZE);
            }
        }

        if (lane_id == 0) y[row] = sum;
    }
}

__device__ inline void dispatch_compute(const BatchInfo batch,
                                        const int * __restrict__ row_ptr,
                                        const int * __restrict__ scol,
                                        const float * __restrict__ sval,
                                        const float * __restrict__ x,
                                        float * __restrict__ y) {
    int row_num = batch.end_row - batch.start_row;
    int nnz_num = batch.end_nnz - batch.start_nnz;
    int mean = row_num > 0 ? (nnz_num / row_num) : 0;

    if      (mean < SWITCH_POINT_1)  vector_compute<1 >(batch, row_ptr, scol, sval, x, y);
    else if (mean < SWITCH_POINT_2)  vector_compute<2 >(batch, row_ptr, scol, sval, x, y);
    else if (mean < SWITCH_POINT_4)  vector_compute<4 >(batch, row_ptr, scol, sval, x, y);
    else if (mean < SWITCH_POINT_8)  vector_compute<8 >(batch, row_ptr, scol, sval, x, y);
    else if (mean < SWITCH_POINT_16) vector_compute<16>(batch, row_ptr, scol, sval, x, y);
    else                             vector_compute<32>(batch, row_ptr, scol, sval, x, y);
}

// =============================================================================
// CSR-Partial-Overlap kernel (Algorithm 4): two-stage pipeline using
// cuda::memcpy_async to overlap global->shared copy of (values, col_idx) for
// the next batch with the SpMV computation of the current batch.
// =============================================================================
__global__ void csr_partial_overlap_kernel(const BatchInfo * __restrict__ batches,
                                           int total_batches,
                                           int block_batch_num,
                                           const int * __restrict__ row_ptr,
                                           const int * __restrict__ col_idx,
                                           const float * __restrict__ values,
                                           const float * __restrict__ x,
                                           float * __restrict__ y) {
    __shared__ float sval[MAX_BATCH_SIZE * 2];
    __shared__ int   scol[MAX_BATCH_SIZE * 2];

    int start_batch = blockIdx.x * block_batch_num;
    int end_batch   = start_batch + block_batch_num;
    if (end_batch > total_batches) end_batch = total_batches;
    if (end_batch <= start_batch) return;

    int my_batch_num = end_batch - start_batch;

    auto block = cg::this_thread_block();
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pss;
    auto pipeline = cuda::make_pipeline(block, &pss);

    const int shared_off[2] = { 0, MAX_BATCH_SIZE };

    int fetch_idx = 0;
    // Prime: fetch first batch.
    {
        BatchInfo b = batches[start_batch + fetch_idx];
        int len = b.end_nnz - b.start_nnz;
        int off = shared_off[fetch_idx & 1];

        pipeline.producer_acquire();
        cuda::memcpy_async(block, sval + off, values    + b.start_nnz,
                           sizeof(float) * len, pipeline);
        cuda::memcpy_async(block, scol + off, col_idx   + b.start_nnz,
                           sizeof(int)   * len, pipeline);
        pipeline.producer_commit();
        fetch_idx++;
    }

    for (int compute_idx = 0; compute_idx < my_batch_num; compute_idx++) {
        // Prefetch the next batch (one stage ahead) if it exists.
        if (fetch_idx < my_batch_num) {
            BatchInfo b = batches[start_batch + fetch_idx];
            int len = b.end_nnz - b.start_nnz;
            int off = shared_off[fetch_idx & 1];

            pipeline.producer_acquire();
            cuda::memcpy_async(block, sval + off, values  + b.start_nnz,
                               sizeof(float) * len, pipeline);
            cuda::memcpy_async(block, scol + off, col_idx + b.start_nnz,
                               sizeof(int)   * len, pipeline);
            pipeline.producer_commit();
            fetch_idx++;
        }

        // Wait for current batch's copy, then compute on it.
        pipeline.consumer_wait();
        block.sync();

        BatchInfo cur = batches[start_batch + compute_idx];
        int off = shared_off[compute_idx & 1];
        dispatch_compute(cur, row_ptr, scol + off, sval + off, x, y);

        block.sync();
        pipeline.consumer_release();
    }
}

// =============================================================================
// Extra-long row kernel (Algorithm 9): one block per long row, multi-warp
// reduction through shared memory.
// =============================================================================
__global__ void long_row_kernel(const int * __restrict__ long_rows,
                                int long_row_num,
                                const int * __restrict__ row_ptr,
                                const int * __restrict__ col_idx,
                                const float * __restrict__ values,
                                const float * __restrict__ x,
                                float * __restrict__ y) {
    __shared__ float warp_sums[32]; // at most 32 warps per block

    int tid     = threadIdx.x;
    int warp_id = tid >> 5;
    int lane    = tid & 31;
    int warps   = blockDim.x >> 5;

    for (int idx = blockIdx.x; idx < long_row_num; idx += gridDim.x) {
        int row = long_rows[idx];
        int nz_start = row_ptr[row];
        int nz_end   = row_ptr[row + 1];

        float sum = 0.0f;
        for (int j = nz_start + tid; j < nz_end; j += blockDim.x)
            sum += values[j] * x[col_idx[j]];

        // Warp-level reduction.
        for (int off = 16; off > 0; off >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, off);

        if (lane == 0) warp_sums[warp_id] = sum;
        __syncthreads();

        if (warp_id == 0) {
            float v = (lane < warps) ? warp_sums[lane] : 0.0f;
            for (int off = 16; off > 0; off >>= 1)
                v += __shfl_down_sync(0xffffffff, v, off);
            if (lane == 0) y[row] = v;
        }
        __syncthreads();
    }
}

// =============================================================================
// Helper: upload CSR matrix to device.
// =============================================================================
static void csr_to_device(const CSR_Matrix *h_mat, CSR_Matrix *d_mat) {
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

int main(void) {
    srand(0);

    DIR *d;
    struct dirent *dir;
    char folder[] = "./matrices/";
    char path[1024];

    d = opendir(folder);
    if (d == NULL) {
        printf("Error opening directory\n");
        return 1;
    }

    // Discover SM count once: use as the number of CUDA blocks (paper sets B
    // to an integer multiple of #SMs; here we pick B such that block_batch_num
    // is >= 1 and load is well distributed).
    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    int sm_count = prop.multiProcessorCount;
    printf("Device: %s, SMs=%d\n", prop.name, sm_count);

    printf("Files in matrices directory:\n");

    TIMER_DEF(0);

    while ((dir = readdir(d)) != NULL) {
        if (dir->d_name[0] == '.') continue;

        const char *ext = strrchr(dir->d_name, '.');
        if (!ext || strcmp(ext, ".mtx") != 0) continue;

        // -------------------------------------------------------
        // MATRIX LOADING

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
        strncpy(stats.implementation, "CUDA CSR-Partial-Overlap",
                sizeof(stats.implementation) - 1);
        stats.rows = csr_A.rows; stats.cols = csr_A.cols; stats.nnz = csr_A.nnz;
        stats.valid = 1;

        // -------------------------------------------------------
        // PREPARATION

        float *x = (float*)malloc((size_t)csr_A.cols * sizeof(*x));
        float *y = (float*)malloc((size_t)csr_A.rows * sizeof(*y));
        if (!x || !y) {
            fprintf(stderr, "Error allocating host vectors\n");
            free_coo(&A); free_csr(&csr_A);
            closedir(d); return 1;
        }
        fill_dense(x, (size_t)csr_A.cols);

        BatchInfo *h_batches = NULL;
        int *h_long_rows = NULL;
        int long_row_count = 0;
        int total_batches = build_batches(&csr_A, &h_batches,
                                          &h_long_rows, &long_row_count);

        int B = sm_count;
        if (B > total_batches) B = total_batches > 0 ? total_batches : 1;
        int block_batch_num = (total_batches + B - 1) / B;

        printf("  CSR-Partial-Overlap: %d batches, %d long rows, "
               "B=%d, block_batch_num=%d\n",
               total_batches, long_row_count, B, block_batch_num);

        // -------------------------------------------------------
        // DEVICE UPLOAD

        float *d_x = NULL, *d_y = NULL;
        cudaMalloc((void**)&d_x, (size_t)csr_A.cols * sizeof(float));
        cudaMalloc((void**)&d_y, (size_t)csr_A.rows * sizeof(float));
        cudaMemcpy(d_x, x, (size_t)csr_A.cols * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemset(d_y, 0, (size_t)csr_A.rows * sizeof(float));

        CSR_Matrix d_csr_A;
        csr_to_device(&csr_A, &d_csr_A);

        BatchInfo *d_batches = NULL;
        cudaMalloc((void**)&d_batches, (size_t)total_batches * sizeof(BatchInfo));
        cudaMemcpy(d_batches, h_batches,
                   (size_t)total_batches * sizeof(BatchInfo),
                   cudaMemcpyHostToDevice);

        int *d_long_rows = NULL;
        if (long_row_count > 0) {
            cudaMalloc((void**)&d_long_rows,
                       (size_t)long_row_count * sizeof(int));
            cudaMemcpy(d_long_rows, h_long_rows,
                       (size_t)long_row_count * sizeof(int),
                       cudaMemcpyHostToDevice);
        }

        // -------------------------------------------------------
        // WARMUP + TIMING

        double times[REPS];

        auto launch = [&]() {
            if (total_batches > 0) {
                csr_partial_overlap_kernel<<<B, THREADS_PER_BLOCK>>>(
                    d_batches, total_batches, block_batch_num,
                    d_csr_A.row_ptr, d_csr_A.col_idx, d_csr_A.values,
                    d_x, d_y);
            }
            if (long_row_count > 0) {
                int lblocks = long_row_count < sm_count ? long_row_count : sm_count;
                long_row_kernel<<<lblocks, 256>>>(
                    d_long_rows, long_row_count,
                    d_csr_A.row_ptr, d_csr_A.col_idx, d_csr_A.values,
                    d_x, d_y);
            }
        };

        for (int r = 0; r < WARMUP; r++) launch();
        cudaDeviceSynchronize();

        for (int r = 0; r < REPS; r++) {
            TIMER_START(0);
            launch();
            cudaDeviceSynchronize();
            TIMER_STOP(0);
            times[r] = TIMER_ELAPSED(0) / 1e6;
        }

        cudaMemcpy(y, d_y, (size_t)csr_A.rows * sizeof(float),
                   cudaMemcpyDeviceToHost);

        // -------------------------------------------------------
        // METRICS

        double total_time = 0.0;
        for (int r = 0; r < REPS; r++) total_time += times[r];
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

        printf("Average time for %s: %.9f s\n", dir->d_name, stats.avg_time_s);
        printf("GFLOP/s for %s: %.6f\n",        dir->d_name, stats.gflops);
        printf("Standard deviation of time for %s: %.9f s\n",
               dir->d_name, stats.std_time_s);
        printf("\n");

        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_batches);
        if (d_long_rows) cudaFree(d_long_rows);
        cudaFree(d_csr_A.row_ptr);
        cudaFree(d_csr_A.col_idx);
        cudaFree(d_csr_A.values);

        free(h_batches);
        free(h_long_rows);
        free_coo(&A);
        free_csr(&csr_A);
        free_dense(x);
        free(y);
    }

    closedir(d);
    return 0;
}
