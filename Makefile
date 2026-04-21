CC = gcc
CFLAGS = -Wall -Wextra -Iinclude
SRC = src/spmv_cpu_single_core.c src/mtx_reader.c src/coo_to_csr.c \
src/generate_dense.c src/csr_spvm.c
OUT = bin/program

all:
	$(CC) $(CFLAGS) $(SRC) -o $(OUT)

clean:
	rm -f $(OUT)