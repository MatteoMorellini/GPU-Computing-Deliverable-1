#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <stdlib.h>
extern "C" {
    #include "mtx_reader.h"
    #include "coo_to_csr.h"
    #include "csr_spvm.h"
}
#include "gpu_common.cuh"
#include <cusparse_v2.h>
#include <cuda.h>

#define CHECK_CUSPARSE(call) do {                                  \
    cusparseStatus_t status = (call);                              \
    if (status != CUSPARSE_STATUS_SUCCESS) {                       \
        fprintf(stderr, "cuSPARSE error at %s:%d: status %d\n",   \
                __FILE__, __LINE__, status);                       \
        exit(EXIT_FAILURE);                                        \
    }                                                             \
} while (0)

int main(void) {
    srand(0);

    // Create cuSPARSE handle once, reuse across all matrices
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    DIR *d;
    struct dirent *dir;
    char folder[] = "./matrices/";
    char path[1024];

    d = opendir(folder);
    if (d == NULL) {
        printf("Error opening directory\n");
        cusparseDestroy(handle);
        return 1;
    }

    printf("Files in matrices directory:\n");

    while ((dir = readdir(d)) != NULL) {
        if (dir->d_name[0] == '.')
            continue;

        char *ext = strrchr(dir->d_name, '.');
        if (!ext || strcmp(ext, ".mtx") != 0)
            continue;

        // -------------------------------------------------------
        // MATRIX LOADING AND CONVERSION SECTION

        snprintf(path, sizeof(path), "%s%s", folder, dir->d_name);

        COO_Matrix A;
        read_mtx(path, &A);
        printf("Read matrix %s: %d rows, %d cols, %d non-zeros\n",
               dir->d_name, A.rows, A.cols, A.nnz);

        CSR_Matrix csr_A;
        coo_to_csr(&A, &csr_A);

        PerfStats stats;
        gpu_init_perf_stats(&stats, dir->d_name, "CSR", "GPU cuSPARSE", &csr_A);

        // -------------------------------------------------------
        // PREPARATION SECTION

        float *x = NULL;
        float *y = NULL;
        if (!gpu_create_host_vectors(&csr_A, &x, &y)) {
            free_coo(&A); free_csr(&csr_A); closedir(d);
            cusparseDestroy(handle);
            return 1;
        }

        // -------------------------------------------------------
        // MOVE TO GPU

        float *d_x = NULL;
        float *d_y = NULL;
        gpu_dense_to_device(&csr_A, x, y, &d_x, &d_y);

        CSR_Matrix d_csr_A;
        gpu_csr_to_device(&csr_A, &d_csr_A);

        // -------------------------------------------------------
        // CUSPARSE DESCRIPTORS

        cusparseSpMatDescr_t matA;
        cusparseDnVecDescr_t vecX, vecY;

        CHECK_CUSPARSE(cusparseCreateCsr(
            &matA,
            d_csr_A.rows, d_csr_A.cols, d_csr_A.nnz,
            d_csr_A.row_ptr,
            d_csr_A.col_idx,
            d_csr_A.values,
            CUSPARSE_INDEX_32I, //row_ptr type is int 32-bit
            CUSPARSE_INDEX_32I, // col_idx type is int 32-bit
            CUSPARSE_INDEX_BASE_ZERO, // zero-based indexing
            CUDA_R_32F // values type is float 32-bit real
        ));

        CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, d_csr_A.cols, d_x, CUDA_R_32F));
        CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, d_csr_A.rows, d_y, CUDA_R_32F));

        // Query and allocate external buffer for SpMV
        float alpha = 1.0f, beta = 0.0f;
        size_t bufferSize = 0;
        void  *dBuffer    = NULL;

        CHECK_CUSPARSE(cusparseSpMV_bufferSize(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, vecX,
            &beta,  vecY,
            CUDA_R_32F,
            CUSPARSE_SPMV_CSR_ALG2,
            &bufferSize
        ));

        CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

        // -------------------------------------------------------
        // WARMUP + TIMING SECTION

        double times[REPS];
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        // Fixed: braces around warmup body so cudaDeviceSynchronize is inside the loop
        for (int r = 0; r < WARMUP; r++) {
            CHECK_CUSPARSE(cusparseSpMV(
                handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, vecX,
                &beta,  vecY,
                CUDA_R_32F,
                CUSPARSE_SPMV_CSR_ALG2,
                dBuffer
            ));
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        for (int r = 0; r < REPS; r++) {
            CHECK_CUDA(cudaEventRecord(start));
            CHECK_CUSPARSE(cusparseSpMV(
                handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, vecX,
                &beta,  vecY,
                CUDA_R_32F,
                CUSPARSE_SPMV_CSR_ALG2,
                dBuffer
            ));
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));

            float elapsed_ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
            times[r] = elapsed_ms / 1e3; // milliseconds -> seconds
        }

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));

        gpu_copy_y_to_host(&csr_A, d_y, y);

        // -------------------------------------------------------
        // PERFORMANCE METRICS SECTION

        gpu_finalize_perf_stats(&stats, times, REPS, csr_A.nnz);
        gpu_print_perf_stats(dir->d_name, &stats);

        // -------------------------------------------------------
        // CLEANUP (per matrix)

        CHECK_CUSPARSE(cusparseDestroySpMat(matA));
        CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
        CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
        CHECK_CUDA(cudaFree(dBuffer));
        gpu_free_dense(d_x, d_y);
        gpu_free_csr(&d_csr_A);

        free_coo(&A);
        free_csr(&csr_A);
        gpu_free_host_vectors(x, y);
    }

    closedir(d);
    CHECK_CUSPARSE(cusparseDestroy(handle));
    return 0;
}
