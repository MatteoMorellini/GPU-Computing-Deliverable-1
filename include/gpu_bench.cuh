#ifndef GPU_BENCH_CUH
#define GPU_BENCH_CUH

#include <cuda_runtime.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
#include "coo_to_csr.h"
#include "generate_dense.h"
#include "mtx_reader.h"
#include "perf_stats.h"
#include "time_lib.h"
}

#include "csr_device.cuh"
#include "stats.cuh"
#include "validate.cuh"

#ifndef BENCH_REPS
#define BENCH_REPS 100
#endif
#ifndef BENCH_WARMUP
#define BENCH_WARMUP 5
#endif
#ifndef BENCH_MATRICES_DIR
#define BENCH_MATRICES_DIR "./matrices/"
#endif

struct BenchConfig {
    const char *implementation; // e.g. "CUDA CSR-Adaptive"
    const char *format;         // e.g. "CSR"
    const char *csv_path;       // e.g. "results/csr_adaptive.csv"
};

// =============================================================================
// run_benchmark: shared per-matrix loop, timing, validation, and CSV output.
//
// Each implementation provides an `Impl` instance whose interface is:
//
//   double prep(const CSR_Matrix &h_csr,
//               const CSR_Matrix &d_csr,
//               float *d_x, float *d_y);
//     Host preprocessing + algorithm-specific device allocations.
//     Returns extra seconds to add to format_conv_s (the harness already
//     times coo_to_csr).
//
//   void launch(const CSR_Matrix &d_csr, const float *d_x, float *d_y);
//     One kernel launch. The harness performs warmup, the timing loop,
//     and synchronization.
//
//   void teardown();
//     Free per-matrix algorithm state.
//
// Setup costs are reported as three separate columns in the CSV:
//   file_parse_s   -> read_mtx
//   format_conv_s  -> coo_to_csr + impl.prep
//   h2d_transfer_s -> csr_to_device (the three CSR arrays only)
// =============================================================================
template <typename Impl>
int run_benchmark(const BenchConfig &cfg, Impl &impl) {
    DIR *d = opendir(BENCH_MATRICES_DIR);
    if (!d) {
        fprintf(stderr, "Error opening directory %s\n", BENCH_MATRICES_DIR);
        return 1;
    }

    FILE *csv = perf_stats_open_csv(perf_stats_resolve_path(cfg.csv_path));
    if (!csv) {
        closedir(d);
        return 1;
    }

    TIMER_DEF(1); // file parse
    TIMER_DEF(2); // format conversion
    TIMER_DEF(3); // H2D transfer of CSR arrays

    printf("Files in %s:\n", BENCH_MATRICES_DIR);

    struct dirent *dir;
    while ((dir = readdir(d)) != NULL) {
        if (dir->d_name[0] == '.') continue;
        const char *ext = strrchr(dir->d_name, '.');
        if (!ext || strcmp(ext, ".mtx") != 0) continue;

        char path[1024];
        snprintf(path, sizeof(path), "%s%s", BENCH_MATRICES_DIR, dir->d_name);

        // ---------------- file parse ----------------
        COO_Matrix A;
        TIMER_START(1);
        read_mtx(path, &A);
        TIMER_STOP(1);
        double file_parse_s = TIMER_ELAPSED(1) / 1e6;
        printf("Read matrix %s: %d rows, %d cols, %d non-zeros\n",
               dir->d_name, A.rows, A.cols, A.nnz);

        // ---------------- COO -> CSR ----------------
        CSR_Matrix csr_A;
        TIMER_START(2);
        coo_to_csr(&A, &csr_A);
        TIMER_STOP(2);
        double format_conv_s = TIMER_ELAPSED(2) / 1e6;

        // ---------------- host vectors ----------------
        float *x = (float*)malloc((size_t)csr_A.cols * sizeof(*x));
        float *y = (float*)malloc((size_t)csr_A.rows * sizeof(*y));
        if (!x || !y) {
            fprintf(stderr, "Error allocating host vectors\n");
            free_coo(&A); free_csr(&csr_A);
            free(x); free(y);
            continue;
        }
        fill_dense(x, (size_t)csr_A.cols);

        // ---------------- device alloc / upload for x, y -----------
        // (vectors are not counted in h2d_transfer_s, per project convention)
        float *d_x = NULL, *d_y = NULL;
        cudaMalloc((void**)&d_x, (size_t)csr_A.cols * sizeof(float));
        cudaMalloc((void**)&d_y, (size_t)csr_A.rows * sizeof(float));
        cudaMemcpy(d_x, x, (size_t)csr_A.cols * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemset(d_y, 0, (size_t)csr_A.rows * sizeof(float));

        // ---------------- H2D transfer of CSR arrays ----------------
        CSR_Matrix d_csr;
        TIMER_START(3);
        csr_to_device(&csr_A, &d_csr);
        cudaDeviceSynchronize();
        TIMER_STOP(3);
        double h2d_transfer_s = TIMER_ELAPSED(3) / 1e6;

        // ---------------- impl-specific preprocessing ----------------
        format_conv_s += impl.prep(csr_A, d_csr, d_x, d_y);

        // ---------------- warmup ----------------
        for (int r = 0; r < BENCH_WARMUP; r++) impl.launch(d_csr, d_x, d_y);
        cudaDeviceSynchronize();

        // ---------------- timed reps (cuda events) ----------------
        cudaEvent_t evt_start, evt_stop;
        cudaEventCreate(&evt_start);
        cudaEventCreate(&evt_stop);

        double times[BENCH_REPS];
        for (int r = 0; r < BENCH_REPS; r++) {
            cudaEventRecord(evt_start);
            impl.launch(d_csr, d_x, d_y);
            cudaEventRecord(evt_stop);
            cudaEventSynchronize(evt_stop);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, evt_start, evt_stop);
            times[r] = (double)ms / 1000.0;
        }
        cudaEventDestroy(evt_start);
        cudaEventDestroy(evt_stop);

        // ---------------- D2H ----------------
        cudaMemcpy(y, d_y, (size_t)csr_A.rows * sizeof(float),
                   cudaMemcpyDeviceToHost);

        // ---------------- stats + validation ----------------
        PerfStats stats;
        memset(&stats, 0, sizeof(stats));
        strncpy(stats.name,           dir->d_name,        sizeof(stats.name) - 1);
        strncpy(stats.format,         cfg.format,         sizeof(stats.format) - 1);
        strncpy(stats.implementation, cfg.implementation, sizeof(stats.implementation) - 1);
        stats.rows  = csr_A.rows;
        stats.cols  = csr_A.cols;
        stats.nnz   = csr_A.nnz;
        stats.valid = 1;

        reduce_kernel_times(times, BENCH_REPS, csr_A.nnz, &stats);
        stats.file_parse_s   = file_parse_s;
        stats.format_conv_s  = format_conv_s;
        stats.h2d_transfer_s = h2d_transfer_s;

        validate_vs_cpu(&csr_A, x, y, &stats);

        printf("Average time for %s: %.9f s\n",            dir->d_name, stats.avg_time_s);
        printf("GFLOP/s for %s: %.6f\n",                   dir->d_name, stats.gflops);
        printf("Standard deviation of time for %s: %.9f s\n", dir->d_name, stats.std_time_s);
        printf("\n");

        perf_stats_write_csv_row(csv, &stats);

        // ---------------- teardown ----------------
        impl.teardown();
        cudaFree(d_x);
        cudaFree(d_y);
        csr_device_free(&d_csr);
        free_coo(&A);
        free_csr(&csr_A);
        free_dense(x);
        free(y);
    }

    closedir(d);
    fclose(csv);
    return 0;
}

// Convenience base for kernels with no per-matrix preprocessing.
struct NoPrepImpl {
    double prep(const CSR_Matrix &, const CSR_Matrix &, float *, float *) {
        return 0.0;
    }
    void teardown() {}
};

#endif // GPU_BENCH_CUH
