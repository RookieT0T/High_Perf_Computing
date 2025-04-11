#include <cuda.h>
#include <stdio.h>
#include <random>
#include "stencil.cuh"

int main(int argc, char* argv[]){
    int n = std::atoi(argv[1]);
    int r = std::atoi(argv[2]);
    int thread_per_block = std::atoi(argv[3]);

    std::mt19937 gen(404);
    std::uniform_real_distribution<float> dist(-1, 1);

    // host array image, output, and mask
    float* imageH = (float*)malloc(n * sizeof(float));
    float* outputH = (float*)malloc(n * sizeof(float));
    float* maskH = (float*)malloc((2 * r + 1) * sizeof(float));
    for(int i = 0; i < n; ++i){
        imageH[i] = dist(gen);
    }
    for(int i = 0; i < (2 * r + 1); ++i){
        maskH[i] = dist(gen);
    }

    // device array image, output, and mask
    float* imageD;
    cudaMalloc((void**)&imageD, n * sizeof(float));
    cudaMemcpy(imageD, imageH, n * sizeof(float), cudaMemcpyHostToDevice);
    float* outputD;
    cudaMalloc((void**)&outputD, n * sizeof(float));
    float* maskD;
    cudaMalloc((void**)&maskD, (2 * r + 1) * sizeof(float));
    cudaMemcpy(maskD, maskH, (2 * r + 1) * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start;
    cudaEvent_t end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    stencil(imageD, maskD, outputD, n, r, thread_per_block);
    cudaEventRecord(end);

    // copy device array back to the host array
    cudaMemcpy(outputH, outputD, n * sizeof(float), cudaMemcpyDeviceToHost);

    float ms;
    cudaEventElapsedTime(&ms, start, end);
    printf("%f\n", outputH[n - 1]);
    printf("%f\n", ms);

    free(imageH);
    free(outputH);
    free(maskH);
    cudaFree(imageD);
    cudaFree(outputD);
    cudaFree(maskD);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return 0;
}