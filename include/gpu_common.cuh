#ifndef GPU_COMMON_CUH
#define GPU_COMMON_CUH

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

extern "C" {
    #include "coo_to_csr.h"
    #include "perf_stats.h"
}

#define REPS 100
#define WARMUP 5

#define CHECK_CUDA(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",              \
                __FILE__, __LINE__, cudaGetErrorString(err));      \
        exit(EXIT_FAILURE);                                        \
    }                                                             \
} while (0)

void gpu_csr_to_device(const CSR_Matrix *h_mat, CSR_Matrix *d_mat);
void gpu_free_csr(CSR_Matrix *d_mat);

int gpu_create_host_vectors(const CSR_Matrix *h_mat, float **h_x, float **h_y);
void gpu_free_host_vectors(float *h_x, float *h_y);

void gpu_dense_to_device(const CSR_Matrix *h_mat,
                         const float *h_x,
                         const float *h_y,
                         float **d_x,
                         float **d_y);
void gpu_copy_y_to_host(const CSR_Matrix *h_mat, const float *d_y, float *h_y);
void gpu_free_dense(float *d_x, float *d_y);

void gpu_init_perf_stats(PerfStats *stats,
                         const char *matrix_name,
                         const char *format,
                         const char *implementation,
                         const CSR_Matrix *csr);
void gpu_finalize_perf_stats(PerfStats *stats,
                             const double *times,
                             int reps,
                             int nnz);
void gpu_print_perf_stats(const char *matrix_name, const PerfStats *stats);

#endif
