#include <cuda.h>
#include <stdio.h>

__global__ void matmul_kernel_int(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim){
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int accu = 0;

    // define share memory
    extern __shared__ int As[block_dim][block_dim];
    extern __shared__ int Bs[block_dim][block_dim];

    int gridDimension = (n + block_dim - 1) / block_dim;
    for(int g = 0; g < gridDimension; ++g){
        // load data into shared memory space
        // check if the element to be accessed is out of matrix
        if(by * block_dim + ty < n && (g * block_dim + tx) < n){
            As[ty][tx] = A[(by * block_dim + ty) * n + g * block_dim + tx];
        }
        else{
            As[ty][tx] = 0;
        }

        if(bx * block_dim + tx < n && (g * block_dim + ty) < n){
            Bs[ty][tx] = B[(g * block_dim + ty) * n + bx * block_dim + tx];
        }
        else{
            Bs[ty][tx] = 0;
        }

        __syncthreads();

        for(int k = 0; k < block_dim; ++k){
            accu += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // update the resultant matrix
    if(by * block_dim + ty < n && bx * block_dim + tx < n){
        C[(by * block_dim + ty) * n + bx * block_dim + tx] = accu;
    }
}

__global__ void matmul_kernel_float(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim){
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float accu = 0;

    // define share memory
    extern __shared__ float As[block_dim][block_dim];
    extern __shared__ float Bs[block_dim][block_dim];

    int gridDimension = (n + block_dim - 1) / block_dim;
    for(int g = 0; g < gridDimension; ++g){
        // load data into shared memory space
        // check if the element to be accessed is out of matrix
        if(by * block_dim + ty < n && (g * block_dim + tx) < n){
            As[ty][tx] = A[(by * block_dim + ty) * n + g * block_dim + tx];
        }
        else{
            As[ty][tx] = 0;
        }

        if(bx * block_dim + tx < n && (g * block_dim + ty) < n){
            Bs[ty][tx] = B[(g * block_dim + ty) * n + bx * block_dim + tx];
        }
        else{
            Bs[ty][tx] = 0;
        }

        __syncthreads();

        for(int k = 0; k < block_dim; ++k){
            accu += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // update the resultant matrix
    if(by * block_dim + ty < n && bx * block_dim + tx < n){
        C[(by * block_dim + ty) * n + bx * block_dim + tx] = accu;
    }
}

__global__ void matmul_kernel_double(const double *A, const double *B, double *C, unsigned int n, unsigned int block_dim){
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    double accu = 0;

    // define share memory
    extern __shared__ double As[block_dim][block_dim];
    extern __shared__ double Bs[block_dim][block_dim];

    int gridDimension = (n + block_dim - 1) / block_dim;
    for(int g = 0; g < gridDimension; ++g){
        // load data into shared memory space
        // check if the element to be accessed is out of matrix
        // row same and col moves
        if(by * block_dim + ty < n && (g * block_dim + tx) < n){
            As[ty][tx] = A[(by * block_dim + ty) * n + g * block_dim + tx];
        }
        else{
            As[ty][tx] = 0;
        }

        // col same and row moves
        if(bx * block_dim + tx < n && (g * block_dim + ty) < n){
            Bs[ty][tx] = B[(g * block_dim + ty) * n + bx * block_dim + tx];
        }
        else{
            Bs[ty][tx] = 0;
        }

        __syncthreads();

        for(int k = 0; k < block_dim; ++k){
            accu += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // update the resultant matrix
    if(by * block_dim + ty < n && bx * block_dim + tx < n){
        C[(by * block_dim + ty) * n + bx * block_dim + tx] = accu;
    }
}

// You should implement Tiled Matrix Multiplication discussed in class
// Computes the matrix product C = AB by making 'one' call to 'matmul_kernel'.
// A, B, and C are row-major representations of nxn matrices in managed memory.
// Configures the kernel call using a 2D configuration with blocks of dimensions
// block_dim x block_dim. The function should end in a call to
// cudaDeviceSynchronize for timing purposes.s
__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim){
    dim3 dimBlock(block_dim, block_dim);
    int safeDimension = (n + block_dim - 1) / block_dim;
    dim3 dimGrid(safeDimension, safeDimension);
    size_t size = sizeof(int) * (block_dim * block_dim);
    matmul_kernel_int<<<dimGrid, dimBlock, size>>>(A, B, C, n, block_dim);
    cudaDeviceSynchronize();
}

__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim){
    dim3 dimBlock(block_dim, block_dim);
    int safeDimension = (n + block_dim - 1) / block_dim;
    dim3 dimGrid(safeDimension, safeDimension);
    size_t size = sizeof(float) * (block_dim * block_dim);
    matmul_kernel_float<<<dimGrid, dimBlock, size>>>(A, B, C, n, block_dim);
    cudaDeviceSynchronize();
}
__host__ void matmul_3(const double *A, const double *B, double *C, unsigned int n, unsigned int block_dim){
    dim3 dimBlock(block_dim, block_dim);
    int safeDimension = (n + block_dim - 1) / block_dim;
    dim3 dimGrid(safeDimension, safeDimension);
    size_t size = sizeof(double) * (block_dim * block_dim);
    matmul_kernel_double<<<dimGrid, dimBlock, size>>>(A, B, C, n, block_dim);
    cudaDeviceSynchronize();
}