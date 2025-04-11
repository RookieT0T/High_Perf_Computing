#include <stdio.h>
#include <cuda.h>

// Computes the matrix product of A and B, storing the result in C.
// Each thread should compute _one_ element of output.
// Does not use shared memory for this problem.
//
// A, B, and C are row major representations of nxn matrices in device memory.
//
// Assumptions:
// - 1D kernel configuration
__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < n * n){
        int row = id / n;   // row for matrix C
        int col = id % n;   // col for matrix C

        float temp = 0;
        for(int i = 0; i < n; ++i){
            float eleA = A[row * n + i];
            float eleB = B[i * n + col];
            temp += eleA * eleB;
        }
        C[row * n + col] = temp;
    }
}

// Makes one call to matmul_kernel with threads_per_block threads per block.
// You can consider following the kernel call with cudaDeviceSynchronize (but if you use 
// cudaEventSynchronize to time it, that call serves the same purpose as cudaDeviceSynchronize).
void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block){
    int numOfElements = n * n;
    int numOfBlocks = (numOfElements + threads_per_block - 1) / threads_per_block;
    matmul_kernel<<<numOfBlocks, threads_per_block>>>(A, B, C, n);
    cudaDeviceSynchronize();
}