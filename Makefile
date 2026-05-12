CC = gcc
NV = nvcc
CFLAGS = -Wall -Wextra -Iinclude
GPU_FLAGS = -Iinclude
SRC = src/spmv_cpu_single_core.c src/mtx_reader.c src/coo_to_csr.c \
	src/generate_dense.c src/csr_spvm.c src/time_lib.c
INFO = src/get_matrix_info.c src/mtx_reader.c src/coo_to_csr.c
GPU_CU  = src/GPU_main.cu
GPU_ADAPTIVE_CU = src/GPU_csr_adaptive.cu
GPU_CUSPARSE_CU = src/GPU_cusparse.cu
GPU_C   = src/mtx_reader.c src/coo_to_csr.c \
	src/generate_dense.c src/csr_spvm.c src/time_lib.c
GPU_OBJS = $(GPU_C:.c=.o)
GPU_COMMON_OBJ = src/gpu_common.o
OUT = bin/program

all:
	$(CC) $(CFLAGS) $(SRC) -o $(OUT) -lm

info:
	$(CC) $(CFLAGS) $(INFO) -o bin/info -lm

# Step 1: compile each .c helper into a .o with gcc
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

src/gpu_common.o: src/gpu_common.cu include/gpu_common.cuh
	$(NV) $(GPU_FLAGS) -c $< -o $@

# Step 2: compile the .cu and link against the .o files
gpu: $(GPU_OBJS) $(GPU_COMMON_OBJ)
	$(NV) $(GPU_FLAGS) $(GPU_CU) $(GPU_OBJS) $(GPU_COMMON_OBJ) -o bin/gpu

adaptive: $(GPU_OBJS) $(GPU_COMMON_OBJ)
	$(NV) $(GPU_FLAGS) $(GPU_ADAPTIVE_CU) $(GPU_OBJS) $(GPU_COMMON_OBJ) -o bin/adaptive

cusparse: $(GPU_OBJS) $(GPU_COMMON_OBJ)
	$(NV) $(GPU_FLAGS) $(GPU_CUSPARSE_CU) $(GPU_OBJS) $(GPU_COMMON_OBJ) -o bin/cusparse -lcusparse

clean:
	rm -f $(OUT) bin/gpu bin/bcsr bin/adaptive bin/dia bin/cusparse bin/info $(GPU_OBJS) $(GPU_COMMON_OBJ)
