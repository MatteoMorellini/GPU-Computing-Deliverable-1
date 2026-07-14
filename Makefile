CC = gcc
NV = nvcc

CFLAGS    = -Wall -Wextra -Iinclude
GPU_FLAGS = -Iinclude
OPENMP_FLAGS = -fopenmp
NV_OPENMP_FLAGS = -Xcompiler -fopenmp

# -----------------------------------------------------------------------------
# Shared helper sources (compiled with gcc, linked into every binary)
# -----------------------------------------------------------------------------
LIB_C_SRC = src/io/mtx_reader.c     \
            src/io/coo_to_csr.c     \
            src/io/generate_dense.c

LIB_C_OBJ = $(LIB_C_SRC:.c=.o)

OPENMP_C_SRC = src/cpu/csr_spvm_openmp.c
OPENMP_C_OBJ = $(OPENMP_C_SRC:.c=.o)
GPU_LIB_OBJ  = $(LIB_C_OBJ) $(OPENMP_C_OBJ)

# -----------------------------------------------------------------------------
# CPU baseline and one-off tools
# -----------------------------------------------------------------------------
CPU_SRC  = src/cpu/spmv_cpu_single_core.c \
           src/cpu/csr_spvm.c \
           $(LIB_C_SRC)
CPU_OPENMP_SRC = src/cpu/spmv_cpu_openmp.c \
                 $(OPENMP_C_SRC) \
                 $(LIB_C_SRC)

# -----------------------------------------------------------------------------
# GPU kernel entrypoints
# -----------------------------------------------------------------------------
KERNEL_DIR = src/kernels

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(OPENMP_C_OBJ): $(OPENMP_C_SRC)
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) -c $< -o $@

all: cpu cpu_openmp

cpu:
	$(CC) $(CFLAGS) $(CPU_SRC) -o bin/program -lm

cpu_openmp:
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) $(CPU_OPENMP_SRC) -o bin/program_openmp -lm

scalar: $(GPU_LIB_OBJ)
	$(NV) $(GPU_FLAGS) $(KERNEL_DIR)/csr_scalar.cu $(GPU_LIB_OBJ) $(NV_OPENMP_FLAGS) -o bin/scalar

vector: $(GPU_LIB_OBJ)
	$(NV) $(GPU_FLAGS) $(KERNEL_DIR)/csr_vector.cu $(GPU_LIB_OBJ) $(NV_OPENMP_FLAGS) -o bin/vector

adaptive: $(GPU_LIB_OBJ)
	$(NV) $(GPU_FLAGS) $(KERNEL_DIR)/csr_adaptive.cu $(GPU_LIB_OBJ) $(NV_OPENMP_FLAGS) -o bin/adaptive

adaptive_paper: $(GPU_LIB_OBJ)
	$(NV) $(GPU_FLAGS) $(KERNEL_DIR)/csr_adaptive_paper.cu $(GPU_LIB_OBJ) $(NV_OPENMP_FLAGS) -o bin/adaptive_paper

partial: $(GPU_LIB_OBJ)
	$(NV) $(GPU_FLAGS) -arch=sm_80 $(KERNEL_DIR)/csr_partial_overlap.cu $(GPU_LIB_OBJ) $(NV_OPENMP_FLAGS) -o bin/partial

cusparse: $(GPU_LIB_OBJ)
	$(NV) $(GPU_FLAGS) $(KERNEL_DIR)/cusparse.cu $(GPU_LIB_OBJ) $(NV_OPENMP_FLAGS) -o bin/cusparse -lcusparse

gpu: scalar vector adaptive adaptive_paper partial cusparse

clean:
	rm -f bin/program bin/program_openmp bin/scalar bin/vector bin/adaptive bin/adaptive_paper bin/partial bin/cusparse $(LIB_C_OBJ) $(OPENMP_C_OBJ) src/cpu/csr_spvm.o src/util/time_lib.o

.PHONY: all cpu cpu_openmp scalar vector adaptive adaptive_paper partial cusparse gpu clean
