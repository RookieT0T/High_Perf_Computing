#include <cuda.h>
#include <stdio.h>
#include <random>
#include "vscale.cuh"

int main(int argc, char* argv[]){
    int n = std::atoi(argv[1]);

    // random generator
    std::mt19937 gen(404);
    std::uniform_real_distribution<float> distA(-10, 10);
    std::uniform_real_distribution<float> distB(0.0, 1.0);

    // host arrays A and B
    float* hA = (float*)malloc(n * sizeof(float));
    float* hB = (float*)malloc(n * sizeof(float));
    for(int i = 0; i < n; ++i){
        hA[i] = distA(gen);
        hB[i] = distB(gen);
    }

    // device array
    float* dA;
    cudaMalloc((void**)&dA, n * sizeof(float));
    cudaMemcpy(dA, hA, n * sizeof(float), cudaMemcpyHostToDevice);
    float* dB;
    cudaMalloc((void**)&dB, n * sizeof(float));
    cudaMemcpy(dB, hB, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // calculate the number of necessary blocks
    const int threadsPerBlock = 512;
    const int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    // timing
    cudaEvent_t start;
    cudaEvent_t end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    vscale<<<numBlocks, threadsPerBlock>>>(dA, dB, n);  // call kernel
    cudaEventRecord(end);
    cudaDeviceSynchronize();

    // copy device array back to the host array
    cudaMemcpy(hB, dB, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // print out results
    float ms;
    cudaEventElapsedTime(&ms, start, end);
    printf("%f\n", ms);
    printf("%f\n", hB[0]);
    printf("%f\n", hB[n - 1]);

    free(hA);
    free(hB);
    cudaFree(dA);
    cudaFree(dB);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return 0;
}