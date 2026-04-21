CC = gcc
CFLAGS = -Wall -Wextra -Iinclude
SRC = src/spmv_cpu_single_core.c src/mtx_reader.c
OUT = bin/program

all:
	$(CC) $(CFLAGS) $(SRC) -o $(OUT)

clean:
	rm -f $(OUT)