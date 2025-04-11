#include <stdio.h>
#include <cuda.h>

// Computes the convolution of image and mask, storing the result in output.
// Each thread should compute _one_ element of the output matrix.
// Shared memory should be allocated _dynamically_ only.
//
// image is an array of length n.
// mask is an array of length (2 * R + 1).
// output is an array of length n.
// All of them are in device memory
//
// Assumptions:
// - 1D configuration
// - blockDim.x >= 2 * R + 1
//
// The following should be stored/computed in shared memory:
// - The entire mask
// - The elements of image that are needed to compute the elements of output corresponding to the threads in the given block
// - The output image elements corresponding to the given block before it is written back to global memory
__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R){
  int threadID = threadIdx.x;
  extern __shared__ float memSpace[];
  float* maskBuffer = memSpace;
  float* imageBuffer = &memSpace[2 * R + 1];
  float* outputBuffer = &memSpace[2 * R + 1 + blockDim.x + 2 * R];

  // let the first 2 * R + 1 threads of every block to populate the buffer in the shared memory space
  if(threadID < (2 * R + 1)){
    maskBuffer[threadID] = mask[threadID];
  }
  __syncthreads();

  // thread may load multiple elements if needed
  for(int i = threadID; i < blockDim.x + 2 * R; i += blockDim.x){
    int index = blockDim.x * blockIdx.x + i - R;
    if(index < 0 || index > n - 1){
      imageBuffer[i] = 1.0f;
    }
    else{
      imageBuffer[i] = image[index];
    }
  }
  __syncthreads();

  // do computations but store computed result in shared memory space
  int finalIndex = blockDim.x * blockIdx.x + threadID;
  if(finalIndex < n){
    float temp = 0.0f;
    for(int j = 0; j <= R * 2; ++j){
      temp += imageBuffer[threadID + j] * maskBuffer[j];
    }
    outputBuffer[threadID] = temp;
  }
  __syncthreads();

  // write to global memory
  if(finalIndex < n){
    output[finalIndex] = outputBuffer[threadID];
  }
}

// Makes one call to stencil_kernel with threads_per_block threads per block.
// You can consider following the kernel call with cudaDeviceSynchronize (but if you use
// cudaEventSynchronize to time it, that call serves the same purpose as cudaDeviceSynchronize).
//
// Assumptions:
// - threads_per_block >= 2 * R + 1
__host__ void stencil(const float* image,
                      const float* mask,
                      float* output,
                      unsigned int n,
                      unsigned int R,
                      unsigned int threads_per_block){

                    int numOfBlocks = (n + threads_per_block - 1) / threads_per_block;
                    size_t size = sizeof(float) * (2 * R + 1 + threads_per_block + 2 * R + threads_per_block);
                    stencil_kernel<<<numOfBlocks, threads_per_block, size>>>(image, mask, output, n, R);
                    cudaDeviceSynchronize();
                    }
