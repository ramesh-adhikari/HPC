#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define N_ROWS 456626 // Number of rows in the matrix
#define N_COLS 456626 // Number of columns in the matrix
#define NNZ 150818 // Number of non-zero entries in the matrix

// // for sample.txt file
// #define N_ROWS 4 // Number of rows in the matrix
// #define N_COLS 6 // Number of columns in the matrix
// #define NNZ 8 // Number of non-zero entries in the matrix

#define TILE_SIZE 500


void read_higgs_twitter_dataset_and_store_in_csr_format(char *filename, int *row_ptr, int *col_ind, int *values, int *vec) {
    char line[N_ROWS];
    int i, j, row, col, value, nonzero_elements = 0;
    FILE *fp = fopen(filename, "r");

    if (!fp) {
        printf("Error opening file %s\n", filename);
        exit(1);
    }
    // Read the first line to get the number of rows and columns in the matrix
    fgets(line, N_ROWS, fp);

    // Initialize the row_ptr array with zeros
    for (i = 0; i < N_ROWS; ++i) row_ptr[i] = 0;

    // Read the rest of the lines and store the non-zero elements
    while (fgets(line, N_ROWS, fp)) {
        fscanf(fp, "%d", &row);
        fscanf(fp, "%d", &col);
        fscanf(fp, "%d", &value);
        col_ind[nonzero_elements] = col;
        values[nonzero_elements] = value;
        row_ptr[row]++;
        nonzero_elements++;
    } 
    // Close the file
    fclose(fp);
    
    // Update the row_ptr array
    for (i = 0; i < N_ROWS; i++) row_ptr[i + 1] += row_ptr[i];

     // initialize  all elements in the dense vector are 1.
    for (i = 0; i < NNZ; i++)
    {
       vec[i]=1;
    }    
    
}

void sparse_matrix_vector_mult(int *row_ptr, int *col_ind, int *values, const int *vec, float *result) {
   
    int i, j,d_v, d_vector[N_ROWS];
    int row_start, row_end, col, val;
   
     // Iterate over all rows in the sparse matrix
    for (i = 0; i < N_ROWS; i++) {
        result[i] = 0; // Initialize result vector with 0
        row_start = row_ptr[i];
        row_end = row_ptr[i + 1];

        // Iterate over all non-zero elements in the current row
        for (j = row_start; j < row_end; j++) {
            col = col_ind[j];
            val =values[j];
            result[i] += val * vec[col];
        }  
    }
}


void sparse_mat_vec_mult_csr_tiling(int *row_ptr, int *col_ind, int *values, const int *vec, float *result) {
    int i, j, row_start, row_end;
    int tile_start, tile_end;
    struct timeval start, end;

    // Iterate over all rows in the sparse matrix
    for (tile_start = 0; tile_start < N_ROWS; tile_start += TILE_SIZE) {
        tile_end = tile_start + TILE_SIZE;
        tile_end = tile_end < N_ROWS ? tile_end : N_ROWS;

        // Iterate over all tiles
        for (i = tile_start; i < tile_end; ++i) {
            // Initialize result vector with 0
            result[i] = 0;
            row_start = row_ptr[i];
            row_end = row_ptr[i + 1];

            // Iterate over all non-zero elements in the current row
            for (j = row_start; j < row_end; ++j) {
                result[i] += values[j] * vec[col_ind[j]];
            }
        }
    }
}


int main() {

     // Allocate memory for CSR matrix
    int *row_ptr = (int*)malloc((N_ROWS + 1) * sizeof(int));
    int *col_ind = (int*)malloc(NNZ * sizeof(int));
    int *values = (int*)malloc(NNZ * sizeof(int));

    // Allocate memory for vector
    int *vec = (int*)malloc(N_COLS * sizeof(int));

    // Allocate memory for result
    float *result = (float*)malloc(N_ROWS * sizeof(float));

    char *filename = "higgs_twitter_mention.mtx";

    // Read the Higgs Twitter dataset
    read_higgs_twitter_dataset_and_store_in_csr_format(filename, row_ptr, col_ind, values, vec);

     // Start timer
    struct timeval start, end;
    gettimeofday(&start, NULL);
    
    // sparse matrix multiplication
    sparse_matrix_vector_mult(row_ptr,col_ind, values, vec, result);

    // sparse matrix multiplication with tiling 
    // sparse_mat_vec_mult_csr_tiling(row_ptr,col_ind, values, vec, result);

     // End timer
    gettimeofday(&end, NULL);
    // int i;
    // for (i = 0; i < N_ROWS; i++) {
    //     printf("%f", result[i]);
    //     printf("\n");
    // } 

     // Print the elapsed time
    printf("Elapsed time:  %ld microseconds\n", ((end.tv_sec - start.tv_sec) * 1000000 + end.tv_usec - start.tv_usec));
    // printf("Elapsed time when TILE_SIZE: %d is %ld microseconds\n",TILE_SIZE, ((end.tv_sec - start.tv_sec) * 1000000 + end.tv_usec - start.tv_usec));


     // Free memory
    free(row_ptr);
    free(col_ind);
    free(values);
    free(vec);
    free(result);

    return 0;
}