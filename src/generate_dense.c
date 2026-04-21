#include <stdio.h>
#include <stdlib.h>
#include "generate_dense.h"

float* generate_dense(float cols){
    float* dense = malloc(cols * sizeof(float));
    if (!dense){
        printf("Error allocating memory for dense matrix\n");
        exit(1);
    }
    srand(0);
    for (int i = 0; i < cols; i++){
        dense[i] = (float)(rand() % 10)+1;
    }
    return dense;
}

void free_dense(float* dense){
    free(dense);
}