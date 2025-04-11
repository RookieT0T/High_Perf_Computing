#include <cuda.h>
#include <stdio.h>
#include <random>
#include "matmul.cuh"

int main(int argc, char* argv[]){
    int n = std::atoi(argv[1]);
    int thread_per_block = std::atoi(argv[2]);

    std::mt19937 gen(404);
    std::uniform_real_distribution<float> dist(-1, 1);

    // host arrays A and B
    float* hA = (float*)malloc(n * n * sizeof(float));
    float* hB = (float*)malloc(n * n * sizeof(float));
    float* hC = (float*)malloc(n * n * sizeof(float));
    for(int i = 0; i < n * n; ++i){
        hA[i] = dist(gen);
        hB[i] = dist(gen);
    }

    // device arrays
    float* dA;
    cudaMalloc((void**)&dA, n * n * sizeof(float));
    cudaMemcpy(dA, hA, n * n * sizeof(float), cudaMemcpyHostToDevice);
    float* dB;
    cudaMalloc((void**)&dB, n * n * sizeof(float));
    cudaMemcpy(dB, hB, n * n * sizeof(float), cudaMemcpyHostToDevice);
    float* dC;
    cudaMalloc((void**)&dC, n * n * sizeof(float));

    cudaEvent_t start;
    cudaEvent_t end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    matmul(dA, dB, dC, n, thread_per_block);
    cudaEventRecord(end);

    // copy device array back to the host array
    cudaMemcpy(hC, dC, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    float ms;
    cudaEventElapsedTime(&ms, start, end);
    printf("%f\n", hC[n * n - 1]);
    printf("%f\n", ms);

    free(hA);
    free(hB);
    free(hC);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return 0;
}