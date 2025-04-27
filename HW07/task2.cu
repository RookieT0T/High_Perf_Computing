#include <cuda.h>
#include <stdio.h>
#include <random>
#include <cmath>
#include "reduce.cuh"

int main(int argc, char* argv[]){
    int n = std::atoi(argv[1]);
    int threads_per_block = std::atoi(argv[2]);

    // prepare the host array
    std::mt19937 gen(404);
    std::uniform_real_distribution<float> dist(-1, 1);
    float* h_input = (float*)malloc(sizeof(float) * n); 
    for(int i = 0; i < n; ++i){
        h_input[i] = dist(gen);
    }

    // prepare device input and output array
    float* d_input;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);
    float* d_output;
    // preallocate d_output large enough to hold the maximum number of blocks needed in the first call
    int outputSize = (n + threads_per_block * 2 - 1) / (threads_per_block * 2); // double loads
    cudaMalloc(&d_output, outputSize * sizeof(float));

    cudaEvent_t start;
    cudaEvent_t end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    reduce(&d_input, &d_output, n, threads_per_block);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float ms;
    cudaEventElapsedTime(&ms, start, end);

    float ans;
    cudaMemcpy(&ans, d_input, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%f\n", ans);
    printf("%f\n", ms);

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}