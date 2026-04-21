#include <stdio.h>
#include <stdlib.h>
#include "mtx_reader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mtx_reader.h"

COO_Matrix read_mtx(const char *filename) {
    FILE *f = fopen(filename, "r");
    char line[1024];
    char object[64], format[64], field[64], symmetry[64];
    COO_Matrix mat;

    if (!f) {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(1);
    }

    if (!fgets(line, sizeof(line), f)) {
        fprintf(stderr, "Error reading header from %s\n", filename);
        exit(1);
    }

    if (sscanf(line, "%%%%MatrixMarket %63s %63s %63s %63s",
               object, format, field, symmetry) != 4) {
        fprintf(stderr, "Invalid Matrix Market header in %s\n", filename);
        exit(1);
    }

    if (strcmp(object, "matrix") != 0 || strcmp(format, "coordinate") != 0) {
        fprintf(stderr, "Only coordinate matrix format is supported\n");
        exit(1);
    }

    do {
        if (!fgets(line, sizeof(line), f)) {
            fprintf(stderr, "Error reading size line from %s\n", filename);
            exit(1);
        }
    } while (line[0] == '%');

    if (sscanf(line, "%d %d %d", &mat.rows, &mat.cols, &mat.nnz) != 3) {
        fprintf(stderr, "Invalid size line in %s\n", filename);
        exit(1);
    }

    mat.row = malloc((size_t)mat.nnz * sizeof(int));
    mat.col = malloc((size_t)mat.nnz * sizeof(int));
    mat.data = malloc((size_t)mat.nnz * sizeof(double));

    if (!mat.row || !mat.col || !mat.data) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    for (int i = 0; i < mat.nnz; i++) {
        int r, c;
        double v;
        int ret;

        if (strcmp(field, "pattern") == 0) {
            ret = fscanf(f, "%d %d", &r, &c);
            if (ret != 2) {
                fprintf(stderr, "Parse error at entry %d\n", i);
                exit(1);
            }
            // for pattern matrices, we assume a value of 1.0 for all non-zero entries
            v = 1.0;
        } else if (strcmp(field, "real") == 0) {
            ret = fscanf(f, "%d %d %lf", &r, &c, &v);
            if (ret != 3) {
                fprintf(stderr, "Parse error at entry %d\n", i);
                exit(1);
            }
        } else {
            fprintf(stderr, "Unsupported Matrix Market field type: %s\n", field);
            exit(1);
        }

        r--;
        c--;

        if (r < 0 || r >= mat.rows || c < 0 || c >= mat.cols) {
            fprintf(stderr, "Invalid index at entry %d: row=%d col=%d\n", i, r, c);
            exit(1);
        }

        mat.row[i] = r;
        mat.col[i] = c;
        mat.data[i] = v;
    }

    fclose(f);
    return mat;
}

void free_coo(COO_Matrix *mat) {
    free(mat->row);
    free(mat->col);
    free(mat->data);
    mat->row = NULL;
    mat->col = NULL;
    mat->data = NULL;
}