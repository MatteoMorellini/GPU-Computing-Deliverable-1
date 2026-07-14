#ifndef VALIDATE_CUH
#define VALIDATE_CUH

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {
#include "coo_to_csr.h"
#include "csr_spvm.h"
#include "perf_stats.h"
}

// Compare a GPU result vector against the multicore CPU baseline. The CPU
// implementation accumulates each row in double precision.
// Updates stats->valid (0 on mismatch) and stats->max_abs_error.
static inline void validate_vs_cpu(const CSR_Matrix *csr,
                                   const float *x,
                                   const float *gpu_y,
                                   PerfStats *stats) {
    double *cpu_y = (double*)malloc((size_t)csr->rows * sizeof(double));
    if (!cpu_y) {
        fprintf(stderr, "Warning: could not allocate cpu_y for correctness check\n");
        stats->valid = 0;
        stats->max_abs_error = INFINITY;
        return;
    }

    spmv_csr_reference_openmp(csr, x, cpu_y);

    int mismatches = 0;
    double max_abs_err = 0.0;
    const double tol = 1e-3;
    for (int i = 0; i < csr->rows; i++) {
        double a = (double)gpu_y[i];
        double b = cpu_y[i];
        double abs_err = fabs(a - b);
        if (abs_err > max_abs_err) max_abs_err = abs_err;
        if (abs_err > tol * fmax(1.0, fabs(b))) {
            if (mismatches < 5)
                printf("  MISMATCH row %d: gpu=%g cpu=%g abs_err=%g\n",
                       i, a, b, abs_err);
            mismatches++;
        }
    }
    if (mismatches) {
        printf("  CORRECTNESS: %d mismatches (max_abs_err=%g)\n",
               mismatches, max_abs_err);
        stats->valid = 0;
    } else {
        printf("  CORRECTNESS: OK\n");
    }
    stats->max_abs_error = max_abs_err;
    free(cpu_y);
}

#endif // VALIDATE_CUH
