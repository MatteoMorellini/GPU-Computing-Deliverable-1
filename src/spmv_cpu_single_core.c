#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include "mtx_reader.h"

int main(){

    DIR *d;
    struct dirent *dir;
    char folder[] = "./matrices/";
    char path[1024];

    d = opendir(folder);
    if(d == NULL){
        printf("Error opening directory\n");
        return 1;
    }

    printf("Files in matrices directory:\n");

    while ((dir = readdir(d)) != NULL){
        if (dir->d_name[0] == '.') continue;

        printf("%s\n", dir->d_name);
        snprintf(path, sizeof(path), "%s%s", folder, dir->d_name);
        COO_Matrix A = read_mtx(path);

        printf("Rows: %d\n", A.rows);
    }
    closedir(d);
    return 0;
}