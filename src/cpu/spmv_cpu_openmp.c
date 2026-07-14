#include <dirent.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "coo_to_csr.h"
#include "csr_spvm.h"
#include "generate_dense.h"
#include "mtx_reader.h"
#include "perf_stats.h"
#include "time_lib.h"

#define REPS 100
#define WARMUP 5
#define VALIDATION_TOLERANCE 1.0e-5f

static int benchmark_matrix(const char *path, const char *name, FILE *csv)
{
    COO_Matrix coo;
    CSR_Matrix csr;
    PerfStats stats = {0};
    double times[REPS];

    read_mtx(path, &coo);
    coo_to_csr(&coo, &csr);

    float *x = malloc((size_t)csr.cols * sizeof(*x));
    float *y = malloc((size_t)csr.rows * sizeof(*y));
    float *reference = malloc((size_t)csr.rows * sizeof(*reference));
    if (!x || !y || !reference) {
        fprintf(stderr, "Error: could not allocate dense vectors for %s\n", name);
        free(x);
        free(y);
        free(reference);
        free_coo(&coo);
        free_csr(&csr);
        return 1;
    }

    fill_dense(x, (size_t)csr.cols);
    spmv_csr(&csr, x, reference);

    for (int repetition = 0; repetition < WARMUP; repetition++)
        spmv_csr_openmp(&csr, x, y);

    TIMER_DEF(0);
    for (int repetition = 0; repetition < REPS; repetition++) {
        TIMER_START(0);
        spmv_csr_openmp(&csr, x, y);
        TIMER_STOP(0);
        times[repetition] = TIMER_ELAPSED(0) / 1.0e6;
    }

    double total_time = 0.0;
    for (int repetition = 0; repetition < REPS; repetition++)
        total_time += times[repetition];

    stats.avg_time_s = total_time / REPS;
    stats.gflops = (2.0 * csr.nnz) / (stats.avg_time_s * 1.0e9);

    double variance = 0.0;
    for (int repetition = 0; repetition < REPS; repetition++) {
        const double difference = times[repetition] - stats.avg_time_s;
        variance += difference * difference;
    }
    stats.std_time_s = sqrt(variance / REPS);

    stats.valid = 1;
    for (int row = 0; row < csr.rows; row++) {
        const double error = fabs((double)y[row] - reference[row]);
        if (error > stats.max_abs_error)
            stats.max_abs_error = error;
        if (error > VALIDATION_TOLERANCE)
            stats.valid = 0;
    }

    snprintf(stats.name, sizeof(stats.name), "%s", name);
    snprintf(stats.format, sizeof(stats.format), "CSR");
    snprintf(stats.implementation, sizeof(stats.implementation), "CPU OpenMP");
    stats.rows = csr.rows;
    stats.cols = csr.cols;
    stats.nnz = csr.nnz;

    printf("%-24s rows=%d nnz=%d time=%.9f s GFLOP/s=%.6f valid=%s\n",
           name, csr.rows, csr.nnz, stats.avg_time_s, stats.gflops,
           stats.valid ? "yes" : "no");
    perf_stats_write_csv_row(csv, &stats);

    free(x);
    free(y);
    free(reference);
    free_coo(&coo);
    free_csr(&csr);
    return stats.valid ? 0 : 1;
}

int main(int argc, char **argv)
{
    const char *folder = argc > 1 ? argv[1] : "./matrices";
    if (argc > 2) {
        fprintf(stderr, "Usage: %s [matrix-directory]\n", argv[0]);
        return 1;
    }

    DIR *directory = opendir(folder);
    if (!directory) {
        fprintf(stderr, "Error: could not open matrix directory '%s'\n", folder);
        return 1;
    }

    FILE *csv = perf_stats_open_csv(
        perf_stats_resolve_path("results/cpu_openmp.csv"));
    if (!csv) {
        closedir(directory);
        return 1;
    }

    srand(0);
    printf("OpenMP CSR SpMV using %d thread(s)\n", omp_get_max_threads());

    int status = 0;
    int matrices = 0;
    struct dirent *entry;
    while ((entry = readdir(directory)) != NULL) {
        const char *extension = strrchr(entry->d_name, '.');
        if (!extension || strcmp(extension, ".mtx") != 0)
            continue;

        char path[4096];
        const int length = snprintf(path, sizeof(path), "%s/%s", folder, entry->d_name);
        if (length < 0 || (size_t)length >= sizeof(path)) {
            fprintf(stderr, "Error: matrix path is too long: %s/%s\n", folder,
                    entry->d_name);
            status = 1;
            continue;
        }

        status |= benchmark_matrix(path, entry->d_name, csv);
        matrices++;
    }

    if (matrices == 0) {
        fprintf(stderr, "Error: no .mtx files found in '%s'\n", folder);
        status = 1;
    }

    fclose(csv);
    closedir(directory);
    return status;
}
