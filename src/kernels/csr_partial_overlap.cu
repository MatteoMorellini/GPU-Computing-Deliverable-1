#include "gpu_bench.cuh"

#include <cuda/pipeline>
#include <cuda/barrier>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;

#define MAX_BATCH_SIZE     1024
#define THREADS_PER_BLOCK  128

#define SWITCH_POINT_1     17
#define SWITCH_POINT_2     34
#define SWITCH_POINT_4     64
#define SWITCH_POINT_8    122
#define SWITCH_POINT_16   216

typedef struct {
    int start_row;   // inclusive
    int end_row;     // exclusive
    int start_nnz;
    int end_nnz;     // exclusive
} BatchInfo;


static int build_batches(const CSR_Matrix *csr,
                         BatchInfo **batches_out,
                         int **long_rows_out,
                         int *long_row_count_out) {
    int m = csr->rows;

    int *long_rows      = (int*)malloc((size_t)m * sizeof(int));
    BatchInfo *batches  = (BatchInfo*)malloc((size_t)(m + 1) * sizeof(BatchInfo));
    if (!long_rows || !batches) {
        fprintf(stderr, "Error allocating batch arrays\n");
        exit(1);
    }

    int long_count    = 0;
    int batch_count   = 0;
    int cur_start_row = -1;
    int cur_nnz_sum   = 0;

    for (int row = 0; row < m; row++) {
        int row_nnz = csr->row_ptr[row + 1] - csr->row_ptr[row];

        if (row_nnz > MAX_BATCH_SIZE) {
            if (cur_start_row >= 0) {
                batches[batch_count].start_row = cur_start_row;
                batches[batch_count].end_row   = row;
                batches[batch_count].start_nnz = csr->row_ptr[cur_start_row];
                batches[batch_count].end_nnz   = csr->row_ptr[row];
                batch_count++;
                cur_start_row = -1;
                cur_nnz_sum   = 0;
            }
            long_rows[long_count++] = row;
            continue;
        }

        if (cur_start_row < 0) {
            cur_start_row = row;
            cur_nnz_sum   = row_nnz;
        } else if (cur_nnz_sum + row_nnz > MAX_BATCH_SIZE) {
            batches[batch_count].start_row = cur_start_row;
            batches[batch_count].end_row   = row;
            batches[batch_count].start_nnz = csr->row_ptr[cur_start_row];
            batches[batch_count].end_nnz   = csr->row_ptr[row];
            batch_count++;
            cur_start_row = row;
            cur_nnz_sum   = row_nnz;
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

    *batches_out        = batches;
    *long_rows_out      = long_rows;
    *long_row_count_out = long_count;
    return batch_count;
}

// =============================================================================
// Device-side compute (Algorithm 5): VECTOR_SIZE threads reduce one row.
// =============================================================================
template <int VECTOR_SIZE>
__device__ inline void vector_compute(const BatchInfo batch,
                                      const int * __restrict__ row_ptr,
                                      const int * __restrict__ scol,
                                      const float * __restrict__ sval,
                                      const float * __restrict__ x,
                                      float * __restrict__ y) {
    int tid        = threadIdx.x;
    int vector_id  = tid / VECTOR_SIZE;
    int vector_num = blockDim.x / VECTOR_SIZE;
    int lane_id    = tid & (VECTOR_SIZE - 1);

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
    int mean    = row_num > 0 ? (nnz_num / row_num) : 0;

    if      (mean < SWITCH_POINT_1)  vector_compute<1 >(batch, row_ptr, scol, sval, x, y);
    else if (mean < SWITCH_POINT_2)  vector_compute<2 >(batch, row_ptr, scol, sval, x, y);
    else if (mean < SWITCH_POINT_4)  vector_compute<4 >(batch, row_ptr, scol, sval, x, y);
    else if (mean < SWITCH_POINT_8)  vector_compute<8 >(batch, row_ptr, scol, sval, x, y);
    else if (mean < SWITCH_POINT_16) vector_compute<16>(batch, row_ptr, scol, sval, x, y);
    else                             vector_compute<32>(batch, row_ptr, scol, sval, x, y);
}


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
    if (end_batch <= start_batch)  return;

    int my_batch_num = end_batch - start_batch;

    auto block = cg::this_thread_block();
    alignas(cuda::pipeline_shared_state<cuda::thread_scope_block, 2>) 
        __shared__ unsigned char pss_storage[sizeof(cuda::pipeline_shared_state<cuda::thread_scope_block, 2>)];
    auto pss = reinterpret_cast<cuda::pipeline_shared_state<cuda::thread_scope_block, 2>*>(pss_storage);
    if (threadIdx.x == 0) new (pss) cuda::pipeline_shared_state<cuda::thread_scope_block, 2>();
    __syncthreads();
    auto pipeline = cuda::make_pipeline(block, pss);

    const int shared_off[2] = { 0, MAX_BATCH_SIZE };

    int fetch_idx = 0;
    {
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

    for (int compute_idx = 0; compute_idx < my_batch_num; compute_idx++) {
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
// Extra-long-row kernel (Algorithm 9): one block per long row, multi-warp
// reduction through shared memory.
// =============================================================================
__global__ void long_row_kernel(const int * __restrict__ long_rows,
                                int long_row_num,
                                const int * __restrict__ row_ptr,
                                const int * __restrict__ col_idx,
                                const float * __restrict__ values,
                                const float * __restrict__ x,
                                float * __restrict__ y) {
    __shared__ float warp_sums[32];

    int tid     = threadIdx.x;
    int warp_id = tid >> 5;
    int lane    = tid & 31;
    int warps   = blockDim.x >> 5;

    for (int idx = blockIdx.x; idx < long_row_num; idx += gridDim.x) {
        int row      = long_rows[idx];
        int nz_start = row_ptr[row];
        int nz_end   = row_ptr[row + 1];

        float sum = 0.0f;
        for (int j = nz_start + tid; j < nz_end; j += blockDim.x)
            sum += values[j] * x[col_idx[j]];

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

struct CsrPartialOverlapImpl {
    BatchInfo *d_batches    = nullptr;
    int       *d_long_rows  = nullptr;
    int total_batches       = 0;
    int long_row_count      = 0;
    int block_batch_num     = 0;
    int B                   = 0;
    int sm_count            = 0;

    CsrPartialOverlapImpl() {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        sm_count = prop.multiProcessorCount;
        printf("Device: %s, SMs=%d\n", prop.name, sm_count);
    }

    double prep(const CSR_Matrix &h_csr, const CSR_Matrix &,
                float *, float *) {
        TIMER_DEF(t);
        TIMER_START(t);
        BatchInfo *h_batches = nullptr;
        int       *h_long   = nullptr;
        total_batches = build_batches(&h_csr, &h_batches, &h_long, &long_row_count);

        // SpMV is DRAM-bound in both pipeline stages, so the in-block
        // memcpy_async/compute overlap caps out at well below 2x. On A30 the
        // 16 KB of shared memory per block limits occupancy to 2 resident
        // blocks/SM, so latency hiding has to come from inter-block
        // parallelism: many short-lived blocks let the scheduler swap fresh
        // ones in while others stall on global memory. We therefore pick a
        // small block_batch_num (just enough to amortize block launch and
        // start the pipeline) and let B grow large. Floor at 2*sm_count so
        // tiny matrices still saturate the device.
        const int TARGET_BBN = 4;
        B = (total_batches + TARGET_BBN - 1) / TARGET_BBN;
        const int B_FLOOR = 2 * sm_count;
        if (B < B_FLOOR)       B = B_FLOOR;
        if (B > total_batches) B = total_batches;
        if (B < 1)             B = 1;
        block_batch_num = total_batches > 0 ? (total_batches + B - 1) / B : 0;

        if (total_batches > 0) {
            cudaMalloc((void**)&d_batches,
                       (size_t)total_batches * sizeof(BatchInfo));
            cudaMemcpy(d_batches, h_batches,
                       (size_t)total_batches * sizeof(BatchInfo),
                       cudaMemcpyHostToDevice);
        }
        if (long_row_count > 0) {
            cudaMalloc((void**)&d_long_rows,
                       (size_t)long_row_count * sizeof(int));
            cudaMemcpy(d_long_rows, h_long,
                       (size_t)long_row_count * sizeof(int),
                       cudaMemcpyHostToDevice);
        }
        free(h_batches);
        free(h_long);
        cudaDeviceSynchronize();
        TIMER_STOP(t);

        printf("  CSR-Partial-Overlap: %d batches, %d long rows, "
               "B=%d, block_batch_num=%d\n",
               total_batches, long_row_count, B, block_batch_num);
        return TIMER_ELAPSED(t) / 1e6;
    }

    void launch(const CSR_Matrix &d_csr, const float *d_x, float *d_y) {
        if (total_batches > 0) {
            csr_partial_overlap_kernel<<<B, THREADS_PER_BLOCK>>>(
                d_batches, total_batches, block_batch_num,
                d_csr.row_ptr, d_csr.col_idx, d_csr.values,
                d_x, d_y);
        }
        if (long_row_count > 0) {
            int lblocks = long_row_count < sm_count ? long_row_count : sm_count;
            long_row_kernel<<<lblocks, 256>>>(
                d_long_rows, long_row_count,
                d_csr.row_ptr, d_csr.col_idx, d_csr.values,
                d_x, d_y);
        }
    }

    void teardown() {
        if (d_batches)   { cudaFree(d_batches);   d_batches   = nullptr; }
        if (d_long_rows) { cudaFree(d_long_rows); d_long_rows = nullptr; }
        total_batches = long_row_count = block_batch_num = B = 0;
    }
};

int main(void) {
    srand(0);
    CsrPartialOverlapImpl impl;
    BenchConfig cfg{"CUDA CSR-Partial-Overlap", "CSR",
                    "results/csr_partial_overlap.csv"};
    return run_benchmark(cfg, impl);
}