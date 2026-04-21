#include <stdio.h>
#include "mtx_reader.h"

int main(){
    COO_Matrix A = read_mtx("./matrices/bone010.mtx");
    printf("Rows: %d\n", A.rows);
    return 0;
}