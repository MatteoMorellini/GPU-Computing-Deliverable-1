#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    printf("Devices found: %d\n\n", device_count);

    for (int dev = 0; dev < device_count; dev++) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, dev);

        printf("Device %d: %s\n", dev, p.name);
        printf("  Compute capability : %d.%d\n", p.major, p.minor);
        printf("  Global memory      : %.0f MB\n", (double)p.totalGlobalMem / (1 << 20));
        printf("  Multiprocessors    : %d\n", p.multiProcessorCount);
        printf("  Clock rate         : %.2f GHz\n", p.clockRate * 1e-6);
        printf("  Warp size          : %d\n\n", p.warpSize);
    }

    return 0;
}
