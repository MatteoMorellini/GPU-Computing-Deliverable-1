#include <math.h>
#include <string.h>

#include "gpu_common.cuh"

extern "C" {
    #include "generate_dense.h"
}

void gpu_csr_to_device(const CSR_Matrix *h_mat, CSR_Matrix *d_mat) {
    d_mat->rows = h_mat->rows;
    d_mat->cols = h_mat->cols;
    d_mat->nnz  = h_mat->nnz;

    d_mat->row_ptr = NULL;
    d_mat->col_idx = NULL;
    d_mat->values  = NULL;

    CHECK_CUDA(cudaMalloc((void**)&d_mat->row_ptr,
                          (size_t)(h_mat->rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_mat->col_idx,
                          (size_t)h_mat->nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_mat->values,
                          (size_t)h_mat->nnz * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_mat->row_ptr, h_mat->row_ptr,
                          (size_t)(h_mat->rows + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mat->col_idx, h_mat->col_idx,
                          (size_t)h_mat->nnz * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mat->values, h_mat->values,
                          (size_t)h_mat->nnz * sizeof(float),
                          cudaMemcpyHostToDevice));
}

void gpu_free_csr(CSR_Matrix *d_mat) {
    CHECK_CUDA(cudaFree(d_mat->row_ptr));
    CHECK_CUDA(cudaFree(d_mat->col_idx));
    CHECK_CUDA(cudaFree(d_mat->values));

    d_mat->row_ptr = NULL;
    d_mat->col_idx = NULL;
    d_mat->values  = NULL;
}

int gpu_create_host_vectors(const CSR_Matrix *h_mat, float **h_x, float **h_y) {
    *h_x = NULL;
    *h_y = NULL;

    *h_x = (float*)malloc((size_t)h_mat->cols * sizeof(**h_x));
    if (*h_x == NULL) {
        fprintf(stderr, "Error allocating memory for dense vector\n");
        return 0;
    }
    fill_dense(*h_x, (size_t)h_mat->cols);

    *h_y = (float*)malloc((size_t)h_mat->rows * sizeof(**h_y));
    if (*h_y == NULL) {
        fprintf(stderr, "Error: could not allocate output vector y\n");
        free_dense(*h_x);
        *h_x = NULL;
        return 0;
    }

    return 1;
}

void gpu_free_host_vectors(float *h_x, float *h_y) {
    free_dense(h_x);
    free(h_y);
}

void gpu_dense_to_device(const CSR_Matrix *h_mat,
                         const float *h_x,
                         const float *h_y,
                         float **d_x,
                         float **d_y) {
    *d_x = NULL;
    *d_y = NULL;

    CHECK_CUDA(cudaMalloc((void**)d_x, (size_t)h_mat->cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)d_y, (size_t)h_mat->rows * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(*d_x, h_x, (size_t)h_mat->cols * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(*d_y, h_y, (size_t)h_mat->rows * sizeof(float),
                          cudaMemcpyHostToDevice));
}

void gpu_copy_y_to_host(const CSR_Matrix *h_mat, const float *d_y, float *h_y) {
    CHECK_CUDA(cudaMemcpy(h_y, d_y, (size_t)h_mat->rows * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

void gpu_free_dense(float *d_x, float *d_y) {
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
}

void gpu_init_perf_stats(PerfStats *stats,
                         const char *matrix_name,
                         const char *format,
                         const char *implementation,
                         const CSR_Matrix *csr) {
    memset(stats, 0, sizeof(*stats));

    strncpy(stats->name, matrix_name, sizeof(stats->name) - 1);
    strncpy(stats->format, format, sizeof(stats->format) - 1);
    strncpy(stats->implementation, implementation,
            sizeof(stats->implementation) - 1);

    stats->rows  = csr->rows;
    stats->cols  = csr->cols;
    stats->nnz   = csr->nnz;
    stats->valid = 1;
}

void gpu_finalize_perf_stats(PerfStats *stats,
                             const double *times,
                             int reps,
                             int nnz) {
    double total_time = 0.0;
    for (int r = 0; r < reps; r++)
        total_time += times[r];

    double avg_time = total_time / reps;
    double variance = 0.0;
    for (int r = 0; r < reps; r++) {
        double diff = times[r] - avg_time;
        variance += diff * diff;
    }
    variance /= reps;

    stats->avg_time_s = avg_time;
    stats->std_time_s = sqrt(variance);
    stats->gflops     = (2.0 * nnz) / (avg_time * 1e9);
}

void gpu_print_perf_stats(const char *matrix_name, const PerfStats *stats) {
    printf("Average time for %s: %.9f s\n", matrix_name, stats->avg_time_s);
    printf("GFLOP/s for %s: %.6f\n", matrix_name, stats->gflops);
    printf("Standard deviation of time for %s: %.9f s\n",
           matrix_name, stats->std_time_s);
    printf("\n");
}
