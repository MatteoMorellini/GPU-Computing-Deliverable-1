CC = gcc
CFLAGS = -Wall -Wextra -Iinclude
SRC = src/spmv_cpu_single_core.c src/mtx_reader.c src/coo_to_csr.c \
src/generate_dense.c src/csr_spvm.c src/time_lib.c
INFO = src/get_matrix_info.c src/mtx_reader.c src/coo_to_csr.c
OUT = bin/program

all:
	$(CC) $(CFLAGS) $(SRC) -o $(OUT)
	
info: 
	$(CC) $(CFLAGS) $(INFO) -o bin/info

clean:
	rm -f $(OUT)