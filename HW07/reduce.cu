#include <cuda.h>
#include <stdio.h>

// implements the 'first add during global load' version (Kernel 4) for the
// parallel reduction g_idata is the array to be reduced, and is available on
// the device. g_odata is the array that the reduced results will be written to,
// and is available on the device. expects a 1D configuration. uses only
// dynamically allocated shared memory.
__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n){
    extern __shared__ float sharedMem[];

    unsigned int tid = threadIdx.x;
    unsigned int firstLoadIdx = blockIdx.x * (blockDim.x * 2) + tid;
    unsigned int secondLoadIdx = firstLoadIdx + blockDim.x;

    if(firstLoadIdx < n){
        sharedMem[tid] = g_idata[firstLoadIdx];
        if(secondLoadIdx < n){
            sharedMem[tid] += g_idata[secondLoadIdx];
        }
    }
    else{
        sharedMem[tid] = 0;
    }

    __syncthreads();        // load finishes

    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
        if(tid < s){
            sharedMem[tid] += sharedMem[tid + s];
        }
        __syncthreads();
    }

    // after the previous for loop reduction, the sum of all floats assigned to this thread block will be in sdata[0]
    if (tid == 0) {
        g_odata[blockIdx.x] = sharedMem[0];     // save this block's partial sum to global memory
    }
}

// the sum of all elements in the *input array should be written to the first
// element of the *input array. calls reduce_kernel repeatedly if needed. _No
// part_ of the sum should be computed on host. *input is an array of length N
// in device memory. *output is an array of length = (number of blocks needed
// for the first call of the reduce_kernel) in device memory. configures the
// kernel calls using threads_per_block threads per block. the function should
// end in a call to cudaDeviceSynchronize for timing purposes
__host__ void reduce(float **input, float **output, unsigned int N, unsigned int threads_per_block){
    float *d_input = *input;
    float *d_output = *output;

    unsigned currentSize = N;
    unsigned currentBlocks = (currentSize + (threads_per_block * 2 - 1)) / (threads_per_block * 2);

    while(currentSize > 1){
        size_t size = sizeof(float) * threads_per_block;        // define the share memory size for each block
        reduce_kernel<<<currentBlocks, threads_per_block, size>>>(d_input, d_output, currentSize);
        cudaDeviceSynchronize();

        currentSize = currentBlocks;        // because of reducetion, number of blocks is the new current size to be reduced
        currentBlocks = (currentSize + (threads_per_block * 2 - 1)) / (threads_per_block * 2);
        
        // swap
        float* temp = d_input;
        d_input = d_output;                 // output of this round of reduction will be the input of next round of reduction
        d_output = temp;
    }
    *input = d_input;
    cudaDeviceSynchronize();
}