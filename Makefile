CC = gcc
NV = nvcc

CFLAGS    = -Wall -Wextra -Iinclude
GPU_FLAGS = -Iinclude

# -----------------------------------------------------------------------------
# Shared helper sources (compiled with gcc, linked into every binary)
# -----------------------------------------------------------------------------
LIB_C_SRC = src/io/mtx_reader.c     \
            src/io/coo_to_csr.c     \
            src/io/generate_dense.c \
            src/cpu/csr_spvm.c      \
            src/util/time_lib.c

LIB_C_OBJ = $(LIB_C_SRC:.c=.o)

# -----------------------------------------------------------------------------
# CPU baseline and one-off tools
# -----------------------------------------------------------------------------
CPU_SRC  = src/cpu/spmv_cpu_single_core.c $(LIB_C_SRC)
INFO_SRC = src/tools/get_matrix_info.c src/io/mtx_reader.c src/io/coo_to_csr.c

# -----------------------------------------------------------------------------
# GPU kernel entrypoints
# -----------------------------------------------------------------------------
KERNEL_DIR = src/kernels

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

all: cpu

cpu:
	$(CC) $(CFLAGS) $(CPU_SRC) -o bin/program -lm

info:
	$(CC) $(CFLAGS) $(INFO_SRC) -o bin/info -lm

scalar: $(LIB_C_OBJ)
	$(NV) $(GPU_FLAGS) $(KERNEL_DIR)/csr_scalar.cu $(LIB_C_OBJ) -o bin/scalar

vector: $(LIB_C_OBJ)
	$(NV) $(GPU_FLAGS) $(KERNEL_DIR)/csr_vector.cu $(LIB_C_OBJ) -o bin/vector

adaptive: $(LIB_C_OBJ)
	$(NV) $(GPU_FLAGS) $(KERNEL_DIR)/csr_adaptive.cu $(LIB_C_OBJ) -o bin/adaptive

partial: $(LIB_C_OBJ)
	$(NV) $(GPU_FLAGS) -arch=sm_80 $(KERNEL_DIR)/csr_partial_overlap.cu $(LIB_C_OBJ) -o bin/partial

partial_tune: $(LIB_C_OBJ)
	$(NV) $(GPU_FLAGS) -arch=sm_80 $(KERNEL_DIR)/csr_partial_overlap_tune.cu $(LIB_C_OBJ) -o bin/partial_tune

cusparse: $(LIB_C_OBJ)
	$(NV) $(GPU_FLAGS) $(KERNEL_DIR)/cusparse.cu $(LIB_C_OBJ) -o bin/cusparse -lcusparse

gpu: scalar vector adaptive partial cusparse

clean:
	rm -f bin/program bin/info bin/scalar bin/vector bin/adaptive bin/partial bin/partial_tune bin/cusparse $(LIB_C_OBJ)

.PHONY: all cpu info scalar vector adaptive partial partial_tune cusparse gpu clean
