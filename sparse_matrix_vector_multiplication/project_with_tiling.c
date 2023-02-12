#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define TILE_SIZE 16

struct CSRMatrix {
    int row_ptr[100];
    int col_ind[100];
    int values[100];
};

void sparse_mat_vec_mult_csr_tiled(struct CSRMatrix mat, int mat_size, int *vec, int vec_size, int *result) {
    int i, j, row_start, row_end;
    int tile_start, tile_end;
    struct timeval start, end;

    // Initialize result vector with 0
    for (i = 0; i < vec_size; ++i) result[i] = 0;

    // Start timer
    gettimeofday(&start, NULL);

    // Iterate over all rows in the sparse matrix
    for (tile_start = 0; tile_start < mat_size; tile_start += TILE_SIZE) {
        tile_end = tile_start + TILE_SIZE;
        tile_end = tile_end < mat_size ? tile_end : mat_size;

        // Iterate over all tiles
        for (i = tile_start; i < tile_end; ++i) {
            row_start = mat.row_ptr[i];
            row_end = mat.row_ptr[i + 1];

            // Iterate over all non-zero elements in the current row
            for (j = row_start; j < row_end; ++j) {
                result[i] += mat.values[j] * vec[mat.col_ind[j]];
            }
        }
    }

    // End timer
    gettimeofday(&end, NULL);

    // Print the elapsed time
    printf("Elapsed time: %ld microseconds\n", ((end.tv_sec - start.tv_sec) * 1000000 + end.tv_usec - start.tv_usec));
}

int main() {
    // Example sparse matrix in CSR format
    struct CSRMatrix mat = {{0, 2, 5, 9}, {0, 2, 1, 2}, {1, 2, 3, 4}};
    int mat_size = 4;
    int vec[] = {1, 2, 3};
    int vec_size = sizeof(vec) / sizeof(vec[0]);
    int result[vec_size];
    int i;

    sparse_mat_vec_mult_csr_tiled(mat, mat_size, vec, vec_size, result);

    for (i = 0; i < vec_size; ++i) printf("%d ", result[i]);
    printf("\n");

    return 0;
}

// Tiling is a technique used to improve the performance of matrix operations by breaking down a large matrix into smaller submatrices, called tiles. This can help reduce the number of cache misses and improve memory access patterns, leading to better performance.

// In this program, we add a tiling step in the outer loop, where we iterate over the rows in tiles of size TILE_SIZE instead of one at a time. This allows us to reduce the number of cache misses and improve