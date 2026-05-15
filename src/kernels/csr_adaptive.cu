#include "gpu_bench.cuh"
#include <vector>
#include <stdint.h>

// -----------------------------------------------------------------------------
// CSR-Adaptive SpMV (Greathouse & Daga, SC'14) with NVIDIA-specific
// optimizations beyond the paper:
//   (1) Parallel per-row reduction (warp-shuffle) when a Stream block holds
//       few but long rows, instead of one thread per row.
//   (2) Three-way adaptive: rows with nnz > NNZ_PER_BLOCK get a dedicated
//       LongRow path where multiple blocks cooperate on the same row via
//       atomicAdd. The paper hands these to single-warp CSR-Vector.
//   (3) Vectorized streaming loads (float4 / int4) for values[] and col_idx[],
//       falling back to scalar when first_nz is not 16B-aligned.
//   (4) __ldg() reads on x[] (read-only cache) and NNZ_PER_BLOCK bumped to
//       the paper-tuned 1024, with thread count decoupled at 256.
// -----------------------------------------------------------------------------

#define NNZ_PER_BLOCK       1024
#define THREADS_PER_BLOCK   256
#define WARP_SIZE           32
#define WARPS_PER_BLOCK     (THREADS_PER_BLOCK / WARP_SIZE)  // 8
#define STREAM_PARALLEL_MAX 8     // <=8 rows -> warp-per-row reduction
#define VECTOR_MAX_ROWS     2    // 1 or 2 rows that fit in LDS -> CSR-Vector

// =============================================================================
// Helpers
// =============================================================================

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

// Stream `nnz` products values[base+i]*x[col_idx[base+i]] into lds[0..nnz).
// Uses float4/int4 loads when the base pointers are 16B-aligned, scalar tail.
__device__ __forceinline__ void stream_to_lds(const float * __restrict__ values,
                                              const int   * __restrict__ col_idx,
                                              const float * __restrict__ x,
                                              float *lds,
                                              int base, int nnz, int tid) {
    const float *vp = values  + base;
    const int   *cp = col_idx + base;
    const bool aligned =
        (reinterpret_cast<uintptr_t>(vp) & 0xF) == 0 &&
        (reinterpret_cast<uintptr_t>(cp) & 0xF) == 0;

    int i = 0;
    if (aligned) {
        const float4 *vp4 = reinterpret_cast<const float4*>(vp);
        const int4   *cp4 = reinterpret_cast<const int4*>(cp);
        const int    nnz4 = nnz >> 2;
        for (int k = tid; k < nnz4; k += THREADS_PER_BLOCK) {
            float4 v = vp4[k];
            int4   c = cp4[k];
            int o = k << 2;
            lds[o + 0] = v.x * __ldg(x + c.x);
            lds[o + 1] = v.y * __ldg(x + c.y);
            lds[o + 2] = v.z * __ldg(x + c.z);
            lds[o + 3] = v.w * __ldg(x + c.w);
        }
        i = nnz4 << 2;
    }
    for (int k = i + tid; k < nnz; k += THREADS_PER_BLOCK)
        lds[k] = vp[k] * __ldg(x + cp[k]);
}

// =============================================================================
// Main kernel: handles Stream and Vector blocks.
// =============================================================================
__global__ void csr_adaptive_kernel(CSR_Matrix mat,
                                    const int * __restrict__ block_row_start,
                                    const int * __restrict__ block_row_end,
                                    const float * __restrict__ x,
                                    float * __restrict__ y) {
    __shared__ float lds[NNZ_PER_BLOCK];

    const int row_start = block_row_start[blockIdx.x];
    const int row_end   = block_row_end[blockIdx.x];
    const int num_rows  = row_end - row_start;
    const int tid       = threadIdx.x;

    if (num_rows > VECTOR_MAX_ROWS) {
        // -------------------- CSR-Stream -------------------------------------
        const int first_nz = mat.row_ptr[row_start];
        const int last_nz  = mat.row_ptr[row_end];
        const int nnz      = last_nz - first_nz;

        stream_to_lds(mat.values, mat.col_idx, x, lds, first_nz, nnz, tid);
        __syncthreads();

        if (num_rows <= STREAM_PARALLEL_MAX) {
            // Optimization (1): one warp per row, parallel reduction from LDS.
            const int warp_id = tid >> 5;
            const int lane    = tid & 31;
            if (warp_id < num_rows) {
                const int row = row_start + warp_id;
                const int ls  = mat.row_ptr[row]     - first_nz;
                const int le  = mat.row_ptr[row + 1] - first_nz;
                float sum = 0.0f;
                for (int j = ls + lane; j < le; j += WARP_SIZE)
                    sum += lds[j];
                sum = warp_reduce_sum(sum);
                if (lane == 0) y[row] = sum;
            }
        } else {
            // Many short rows: scalar per-row reduction. May sweep multiple
            // passes if num_rows exceeds THREADS_PER_BLOCK.
            for (int local_row = tid; local_row < num_rows;
                 local_row += THREADS_PER_BLOCK) {
                const int row = row_start + local_row;
                const int ls  = mat.row_ptr[row]     - first_nz;
                const int le  = mat.row_ptr[row + 1] - first_nz;
                float sum = 0.0f;
                #pragma unroll 4
                for (int j = ls; j < le; j++) sum += lds[j];
                y[row] = sum;
            }
        }
    } else {
        // -------------------- CSR-Vector (1 or 2 rows, all fit in LDS) -------
        const int warp_id = tid >> 5;
        const int lane    = tid & 31;
        if (warp_id < num_rows) {
            const int row = row_start + warp_id;
            const int s   = mat.row_ptr[row];
            const int e   = mat.row_ptr[row + 1];
            float sum = 0.0f;
            for (int j = s + lane; j < e; j += WARP_SIZE)
                sum += mat.values[j] * __ldg(x + mat.col_idx[j]);
            sum = warp_reduce_sum(sum);
            if (lane == 0) y[row] = sum;
        }
    }
}

// =============================================================================
// LongRow kernel: each block reduces one NNZ_PER_BLOCK-sized chunk of a row
// that is too long to fit in LDS, then atomicAdds the partial to y[row].
// =============================================================================
__global__ void csr_longrow_kernel(CSR_Matrix mat,
                                   const int * __restrict__ long_row_idx,
                                   const int * __restrict__ long_chunk_off,
                                   const float * __restrict__ x,
                                   float * __restrict__ y) {
    __shared__ float lds[NNZ_PER_BLOCK];
    __shared__ float warp_partials[WARPS_PER_BLOCK];

    const int chunk = blockIdx.x;
    const int row   = long_row_idx[chunk];
    const int off   = long_chunk_off[chunk];
    const int tid   = threadIdx.x;

    const int row_first = mat.row_ptr[row];
    const int row_last  = mat.row_ptr[row + 1];
    const int chunk_first = row_first + off;
    int chunk_last = chunk_first + NNZ_PER_BLOCK;
    if (chunk_last > row_last) chunk_last = row_last;
    const int nnz = chunk_last - chunk_first;

    stream_to_lds(mat.values, mat.col_idx, x, lds, chunk_first, nnz, tid);
    __syncthreads();

    // Block-wide reduction of lds[0..nnz).
    float sum = 0.0f;
    #pragma unroll 4
    for (int i = tid; i < nnz; i += THREADS_PER_BLOCK)
        sum += lds[i];

    sum = warp_reduce_sum(sum);
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    if (lane == 0) warp_partials[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane < WARPS_PER_BLOCK) ? warp_partials[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) atomicAdd(&y[row], sum);
    }
}

// =============================================================================
// Host-side block builder: produces three arrays.
//   normal_start[B], normal_end[B]    : Stream/Vector blocks (row ranges)
//   long_row[L],     long_off[L]      : LongRow chunks (one per NNZ_PER_BLOCK)
// =============================================================================
struct BlockPlan {
    std::vector<int> nstart, nend;
    std::vector<int> lrow, loff;
};

static void build_block_plan(const CSR_Matrix *csr, BlockPlan &plan) {
    const int R = csr->rows;
    const int *rp = csr->row_ptr;

    int seg_start = 0;       // start row of pending normal segment
    int seg_nnz   = 0;       // nnz accumulated in pending segment
    int last_break = 0;      // row index where the segment last advanced past

    auto flush_segment = [&](int seg_end) {
        if (seg_end > seg_start) {
            plan.nstart.push_back(seg_start);
            plan.nend.push_back(seg_end);
        }
        seg_start  = seg_end;
        seg_nnz    = 0;
        last_break = seg_end;
    };

    int i = 0;
    while (i < R) {
        const int row_nnz = rp[i + 1] - rp[i];

        if (row_nnz > NNZ_PER_BLOCK) {
            // Long row: close any pending segment, then emit chunks.
            flush_segment(i);
            const int chunks = (row_nnz + NNZ_PER_BLOCK - 1) / NNZ_PER_BLOCK;
            for (int c = 0; c < chunks; c++) {
                plan.lrow.push_back(i);
                plan.loff.push_back(c * NNZ_PER_BLOCK);
            }
            i++;
            seg_start  = i;
            last_break = i;
            continue;
        }

        if (seg_nnz + row_nnz > NNZ_PER_BLOCK) {
            // Adding this row would overflow LDS: close segment before row i.
            if (i > last_break) {
                flush_segment(i);
                // Retry this row at the head of a fresh segment.
                continue;
            } else {
                // Single row fits exactly in <= NNZ_PER_BLOCK; emit alone.
                flush_segment(i + 1);
                i++;
                continue;
            }
        }

        seg_nnz += row_nnz;
        i++;
    }
    flush_segment(R);
}

// =============================================================================
// Bench harness implementation
// =============================================================================
struct CsrAdaptiveImpl {
    int *d_nstart = nullptr, *d_nend = nullptr;
    int *d_lrow   = nullptr, *d_loff = nullptr;
    int num_normal = 0, num_long = 0;
    int rows = 0;

    double prep(const CSR_Matrix &h_csr, const CSR_Matrix &,
                float *, float *d_y_unused) {
        (void)d_y_unused;
        TIMER_DEF(t);
        TIMER_START(t);

        BlockPlan plan;
        build_block_plan(&h_csr, plan);
        num_normal = (int)plan.nstart.size();
        num_long   = (int)plan.lrow.size();
        rows       = h_csr.rows;

        cudaMalloc((void**)&d_nstart, (size_t)num_normal * sizeof(int));
        cudaMalloc((void**)&d_nend,   (size_t)num_normal * sizeof(int));
        cudaMemcpy(d_nstart, plan.nstart.data(),
                   num_normal * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nend,   plan.nend.data(),
                   num_normal * sizeof(int), cudaMemcpyHostToDevice);
        if (num_long > 0) {
            cudaMalloc((void**)&d_lrow, (size_t)num_long * sizeof(int));
            cudaMalloc((void**)&d_loff, (size_t)num_long * sizeof(int));
            cudaMemcpy(d_lrow, plan.lrow.data(),
                       num_long * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_loff, plan.loff.data(),
                       num_long * sizeof(int), cudaMemcpyHostToDevice);
        }

        cudaDeviceSynchronize();
        TIMER_STOP(t);
        printf("  CSR-Adaptive: %d normal blocks + %d long-row chunks "
               "(%d nnz/block, %d threads)\n",
               num_normal, num_long, NNZ_PER_BLOCK, THREADS_PER_BLOCK);
        return TIMER_ELAPSED(t) / 1e6;
    }

    void launch(const CSR_Matrix &d_csr, const float *d_x, float *d_y) {
        // LongRow blocks atomicAdd into y[row] -- those entries must be zero
        // at the start of every launch. Stream/Vector blocks overwrite their
        // own rows, so zeroing the full y[] is sufficient and cheap.
        if (num_long > 0)
            cudaMemsetAsync(d_y, 0, (size_t)rows * sizeof(float));

        if (num_normal > 0)
            csr_adaptive_kernel<<<num_normal, THREADS_PER_BLOCK>>>(
                d_csr, d_nstart, d_nend, d_x, d_y);
        if (num_long > 0)
            csr_longrow_kernel<<<num_long, THREADS_PER_BLOCK>>>(
                d_csr, d_lrow, d_loff, d_x, d_y);
    }

    void teardown() {
        if (d_nstart) cudaFree(d_nstart);
        if (d_nend)   cudaFree(d_nend);
        if (d_lrow)   cudaFree(d_lrow);
        if (d_loff)   cudaFree(d_loff);
        d_nstart = d_nend = d_lrow = d_loff = nullptr;
        num_normal = num_long = 0;
    }
};

int main(void) {
    srand(0);
    CsrAdaptiveImpl impl;
    BenchConfig cfg{"CUDA CSR-Adaptive", "CSR", "results/csr_adaptive.csv"};
    return run_benchmark(cfg, impl);
}
