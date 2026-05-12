#ifndef STATS_CUH
#define STATS_CUH

#include <math.h>

extern "C" {
#include "perf_stats.h"
}

// Reduce a per-iteration array of kernel times (in seconds) into mean,
// standard deviation, and GFLOP/s. The first three PerfStats fields are
// overwritten; the caller still owns matrix identification and setup-cost
// fields.
static inline void reduce_kernel_times(const double *times,
                                       int n,
                                       int nnz,
                                       PerfStats *stats) {
    double total = 0.0;
    for (int r = 0; r < n; r++) total += times[r];
    double avg = total / (double)n;

    double var = 0.0;
    for (int r = 0; r < n; r++) {
        double diff = times[r] - avg;
        var += diff * diff;
    }
    var /= (double)n;

    stats->avg_time_s = avg;
    stats->std_time_s = sqrt(var);
    stats->gflops     = (2.0 * (double)nnz) / (avg * 1e9);
}

#endif // STATS_CUH
