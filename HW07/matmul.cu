#include <cuda.h>
#include <stdio.h>

__global__ void matmul_kernel_int(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim){
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int accu = 0;

    // define share memory
    extern __shared__ int sharedMem1[];
    int* As = sharedMem1;
    int* Bs = &sharedMem1[block_dim * block_dim];

    int gridDimension = (n + block_dim - 1) / block_dim;
    for(int g = 0; g < gridDimension; ++g){
        // load data into shared memory space
        // check if the element to be accessed is out of matrix
        if(by * block_dim + ty < n && (g * block_dim + tx) < n){
            As[ty * block_dim + tx] = A[(by * block_dim + ty) * n + g * block_dim + tx];
        }
        else{
            As[ty * block_dim + tx] = 0;
        }

        if(bx * block_dim + tx < n && (g * block_dim + ty) < n){
            Bs[ty * block_dim + tx] = B[(g * block_dim + ty) * n + bx * block_dim + tx];
        }
        else{
            Bs[ty * block_dim + tx] = 0;
        }

        __syncthreads();

        for(int k = 0; k < block_dim; ++k){
            accu += As[ty * block_dim + k] * Bs[k * block_dim + tx];
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

    float accu = 0.0f;

    // define share memory
    extern __shared__ float sharedMem2[];
    float* As = sharedMem2;
    float* Bs = &sharedMem2[block_dim * block_dim];

    int gridDimension = (n + block_dim - 1) / block_dim;
    for(int g = 0; g < gridDimension; ++g){
        // load data into shared memory space
        // check if the element to be accessed is out of matrix
        if(by * block_dim + ty < n && (g * block_dim + tx) < n){
            As[ty * block_dim + tx] = A[(by * block_dim + ty) * n + g * block_dim + tx];
        }
        else{
            As[ty * block_dim + tx] = 0.0f;
        }

        if(bx * block_dim + tx < n && (g * block_dim + ty) < n){
            Bs[ty * block_dim + tx] = B[(g * block_dim + ty) * n + bx * block_dim + tx];
        }
        else{
            Bs[ty * block_dim + tx] = 0.0f;
        }

        __syncthreads();

        for(int k = 0; k < block_dim; ++k){
            accu += As[ty * block_dim + k] * Bs[k * block_dim + tx];
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

    double accu = 0.0f;

    // define share memory
    extern __shared__ double sharedMem3[];
    double* As = sharedMem3;
    double* Bs = &sharedMem3[block_dim * block_dim];

    int gridDimension = (n + block_dim - 1) / block_dim;
    for(int g = 0; g < gridDimension; ++g){
        // load data into shared memory space
        // check if the element to be accessed is out of matrix
        // row same and col moves
        if(by * block_dim + ty < n && (g * block_dim + tx) < n){
            As[ty * block_dim + tx] = A[(by * block_dim + ty) * n + g * block_dim + tx];
        }
        else{
            As[ty * block_dim + tx] = 0.0f;
        }

        // col same and row moves
        if(bx * block_dim + tx < n && (g * block_dim + ty) < n){
            Bs[ty * block_dim + tx] = B[(g * block_dim + ty) * n + bx * block_dim + tx];
        }
        else{
            Bs[ty * block_dim + tx] = 0.0f;
        }

        __syncthreads();

        for(int k = 0; k < block_dim; ++k){
            accu += As[ty * block_dim + k] * Bs[k * block_dim + tx];
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
    size_t size = 2 * sizeof(int) * (block_dim * block_dim);
    matmul_kernel_int<<<dimGrid, dimBlock, size>>>(A, B, C, n, block_dim);
    cudaDeviceSynchronize();
}

__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim){
    dim3 dimBlock(block_dim, block_dim);
    int safeDimension = (n + block_dim - 1) / block_dim;
    dim3 dimGrid(safeDimension, safeDimension);
    size_t size = 2 * sizeof(float) * (block_dim * block_dim);
    matmul_kernel_float<<<dimGrid, dimBlock, size>>>(A, B, C, n, block_dim);
    cudaDeviceSynchronize();
}
__host__ void matmul_3(const double *A, const double *B, double *C, unsigned int n, unsigned int block_dim){
    dim3 dimBlock(block_dim, block_dim);
    int safeDimension = (n + block_dim - 1) / block_dim;
    dim3 dimGrid(safeDimension, safeDimension);
    size_t size = 2 * sizeof(double) * (block_dim * block_dim);
    matmul_kernel_double<<<dimGrid, dimBlock, size>>>(A, B, C, n, block_dim);
    cudaDeviceSynchronize();
}