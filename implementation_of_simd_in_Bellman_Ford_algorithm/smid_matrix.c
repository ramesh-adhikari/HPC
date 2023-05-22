#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#define N 64 // Size of the input matrices
#define T 8 // Tile size for shared memory

__global__ void matrixMul(float* A, float* B, float* C) {
    __shared__ float smem_c[N][N];
    __shared__ float smem_a[N][T];
    __shared__ float smem_b[T][N];

    int c = blockIdx.x * N;
    int r = blockIdx.y * N;

    for (int kk = 0; kk < N; kk += T) {
          for(int i=threadIdx.x+blockDim.x*threadIdx.y; i<N*T; i+=blockDim.x*blockDim.y) {
                int k = kk + i/N;
                int rt = r + i%N;
                int ct = c + i%N;

                //Load A[i][j] to shared mem
                smem_a[i%N][i/N] = A[rt * N + k]; //Coalesced access

                //Load B[i][j] to shared mem
                smem_b[i/64][i%N] = B[k * N + ct]; // Coalesced access

        }
        //int k = kk + threadIdx.x;
        //int rt = r + threadIdx.y;
        //int ct = c + threadIdx.x;

        //Load A[i][j] to shared mem
        //smem_a[threadIdx.y][threadIdx.x] = A[rt * N + k]; //Coalesced access

        //Load B[i][j] to shared mem
        //smem_b[threadIdx.y][threadIdx.x] = B[k * N + ct]; // Coalesced access

        //Synchronize before computation
        __syncthreads();

       //Add one tile of result from tiles of A and B in shared mem
        for (int i = 0; i < T; i++) {
            smem_c[threadIdx.y][threadIdx.x] += smem_a[threadIdx.y][i] * smem_b[i][threadIdx.x];
        }

       //Wait for all threads to finish using current tiles before loading in new one
        __syncthreads();
    }

    int rt = r + threadIdx.y;
    int ct = c + threadIdx.x;

    //Store accumalated value to c
    C[rt * N + ct] = smem_c[threadIdx.y][threadIdx.x];
}

int main() {
    // Allocate memory for input and output matrices
    float* A = new float[N * N];
    float* B = new float[N * N];
    float* C = new float[N * N];

    // Initialize input matrices A and B

    // Allocate device memory for input and output matrices
    float* d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float));

    // Copy input matrices A and B from host to device
    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 gridDim(N / T, N / T);
    dim3 blockDim(T, T);
debug2: channel 0: window 999420 sent adjust 49156
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch CUDA kernel for matrix multiplication
    matrixMul<<<gridDim, blockDim>>>(d_A, d_B, d_C);

    // Copy output matrix C from device to host
    cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time: %.3f ms\n", milliseconds);


  // Check for any CUDA errors
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaError));
        return 1;
    }
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Clean up host memory
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
