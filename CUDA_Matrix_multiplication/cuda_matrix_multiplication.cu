#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 64
#define TILE_SIZEB  64
__global__ void matrixMul(float* A, float* B, float* C, int N) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];
   // __shared__ float sC[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float result = 0.0;

    for (int i = 0; i < N/TILE_SIZE; i++) {
        sA[ty][tx] = A[row*N + i*TILE_SIZE + tx];
        sB[ty][tx] = B[(i*TILE_SIZE + ty)*N + col];
        __syncthreads();

        for (int j = 0; j < TILE_SIZE; j++) {
            result += sA[ty][j] * sB[j][tx];
        }
        __syncthreads();
    }

    C[row*N + col] = result;
}

int main() {
    int N = 1024;
    int size = N * N * sizeof(float);

    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

    // Allocate host memory
    h_A = (float*) malloc(size);
    h_B = (float*) malloc(size);
    h_C = (float*) malloc(size);

    // Initialize matrices
    for (int i = 0; i < N*N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Allocate device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy input matrices to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    //dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    //dim3 dimGrid((N+TILE_SIZE-1)/TILE_SIZE, (N+TILE_SIZEB-1)/TILE_SIZEB);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid(N/TILE_SIZE, N/TILE_SIZEB);


    // Copy result matrix back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matrixMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Elapsed time: %f ms\n", milliseconds);

    // Verify result
   /***
    for (int i = 0; i < N*N; i++) {
        float expected = 0.0;
        int row = i / N;
        int col = i % N;
        for (int j = 0; j < N; j++) {
            expected += h_A[row*N + j] * h_B[j*N + col];
        }
        if (abs(h_C[i] - expected) > 1e-5) {
            printf("Mismatch at (%d, %d): expected %f, got %f\n",
                   row, col, expected, h_C[i]);
            exit(1);
        }
    }
    printf("Test passed!\n");
    ***/

    // Check for any CUDA errors
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaError));
        return 1;
    }

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

