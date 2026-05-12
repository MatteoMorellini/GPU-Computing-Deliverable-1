#include <stdio.h>
#include <stdlib.h>
#include "generate_dense.h"

void fill_dense(float *x, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        x[i] = (float)(rand() % 10 + 1);
    }
}

void free_dense(float* dense){
    free(dense);
}