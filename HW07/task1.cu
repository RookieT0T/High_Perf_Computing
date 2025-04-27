#include <cuda.h>
#include <stdio.h>
#include <random>
#include <cmath>
#include "matmul.cuh"

int main(int argc, char* argv[]){
    int n = std::atoi(argv[1]);
    int block_dim = std::atoi(argv[2]);;

    std::mt19937 gen(404);
    std::uniform_real_distribution<float> dist_float(-10, 10);

    // host arrays A and B - int & float & double
    int* hA_int = (int*)malloc(n * n * sizeof(int));
    float* hA_float = (float*)malloc(n * n * sizeof(float));
    double* hA_double = (double*)malloc(n * n * sizeof(double));

    int* hB_int = (int*)malloc(n * n * sizeof(int));
    float* hB_float = (float*)malloc(n * n * sizeof(float));
    double* hB_double = (double*)malloc(n * n * sizeof(double));

    int* hC_int = (int*)malloc(n * n * sizeof(int));
    float* hC_float = (float*)malloc(n * n * sizeof(float));
    double* hC_double = (double*)malloc(n * n * sizeof(double));

    for(int i = 0; i < n * n; ++i){
        float A = dist_float(gen);
        float B = dist_float(gen);

        hA_int[i] = static_cast<int>(A);
        hB_int[i] = static_cast<int>(B);

        hA_float[i] = A;
        hB_float[i] = B;

        hA_double[i] = static_cast<double>(A);
        hB_double[i] = static_cast<double>(B);
    }

    // device arrays A and B - int & float & double
    int* dA_int;
    cudaMalloc((void**)&dA_int, n * n * sizeof(int));
    cudaMemcpy(dA_int, hA_int, n * n * sizeof(int), cudaMemcpyHostToDevice);
    int* dB_int;
    cudaMalloc((void**)&dB_int, n * n * sizeof(int));
    cudaMemcpy(dB_int, hB_int, n * n * sizeof(int), cudaMemcpyHostToDevice);
    int* dC_int;
    cudaMalloc((void**)&dC_int, n * n * sizeof(int));

    float* dA_float;
    cudaMalloc((void**)&dA_float, n * n * sizeof(float));
    cudaMemcpy(dA_float, hA_float, n * n * sizeof(float), cudaMemcpyHostToDevice);
    float* dB_float;
    cudaMalloc((void**)&dB_float, n * n * sizeof(float));
    cudaMemcpy(dB_float, hB_float, n * n * sizeof(float), cudaMemcpyHostToDevice);
    float* dC_float;
    cudaMalloc((void**)&dC_float, n * n * sizeof(float));

    double* dA_double;
    cudaMalloc((void**)&dA_double, n * n * sizeof(double));
    cudaMemcpy(dA_double, hA_double, n * n * sizeof(double), cudaMemcpyHostToDevice);
    double* dB_double;
    cudaMalloc((void**)&dB_double, n * n * sizeof(double));
    cudaMemcpy(dB_double, hB_double, n * n * sizeof(double), cudaMemcpyHostToDevice);
    double* dC_double;
    cudaMalloc((void**)&dC_double, n * n * sizeof(double));

    cudaEvent_t start_int;
    cudaEvent_t end_int;
    cudaEvent_t start_float;
    cudaEvent_t end_float;
    cudaEvent_t start_double;
    cudaEvent_t end_double;
    cudaEventCreate(&start_int);
    cudaEventCreate(&end_int);
    cudaEventCreate(&start_float);
    cudaEventCreate(&end_float);
    cudaEventCreate(&start_double);
    cudaEventCreate(&end_double);

    // call functions
    cudaEventRecord(start_int);
    matmul_1(dA_int, dB_int, dC_int, n, block_dim);
    cudaEventRecord(end_int);
    cudaEventSynchronize(end_int);
    float ms_int;
    cudaEventElapsedTime(&ms_int, start_int, end_int);

    cudaEventRecord(start_float);
    matmul_2(dA_float, dB_float, dC_float, n, block_dim);
    cudaEventRecord(end_float);
    cudaEventSynchronize(end_float);
    float ms_float;
    cudaEventElapsedTime(&ms_float, start_float, end_float);

    cudaEventRecord(start_double);
    matmul_3(dA_double, dB_double, dC_double, n, block_dim);
    cudaEventRecord(end_double);
    cudaEventSynchronize(end_double);
    float ms_double;
    cudaEventElapsedTime(&ms_double, start_double, end_double);

    // copy results from device back to host
    cudaMemcpy(hC_int, dC_int, n * n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hC_float, dC_float, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hC_double, dC_double, n * n * sizeof(double), cudaMemcpyDeviceToHost);

    // print results
    printf("%d\n", hC_int[0]);
    printf("%d\n", hC_int[n * n - 1]);
    printf("%f\n", ms_int);

    printf("%f\n", hC_float[0]);
    printf("%f\n", hC_float[n * n - 1]);
    printf("%f\n", ms_float);

    printf("%f\n", hC_double[0]);
    printf("%f\n", hC_double[n * n - 1]);
    printf("%f\n", ms_double);

    free(hA_int);
    free(hA_float);
    free(hA_double);
    free(hB_int);
    free(hB_float);
    free(hB_double);
    free(hC_int);
    free(hC_float);
    free(hC_double);
    cudaFree(dA_int);
    cudaFree(dB_int);
    cudaFree(dC_int);
    cudaFree(dA_float);
    cudaFree(dB_float);
    cudaFree(dC_float);
    cudaFree(dA_double);
    cudaFree(dB_double);
    cudaFree(dC_double);
    cudaEventDestroy(start_int);
    cudaEventDestroy(end_int);
    cudaEventDestroy(start_float);
    cudaEventDestroy(end_float);
    cudaEventDestroy(start_double);
    cudaEventDestroy(end_double);
    return 0;
}