#include <iostream>
#include <cstdlib>
#include <random>
#include <chrono>
#include <omp.h>
#include "matmul.h"

int main(int argc, char* argv[]){
    int N = std::atoi(argv[1]);
    int T = std::atoi(argv[2]);

    omp_set_num_threads(T);

    // set up random generator
    std::random_device rd; 
    std::mt19937 gen(404);
    std::uniform_real_distribution<float> dist(-10, 10);

    // declare arrays and fill the values
    float* A = (float*)malloc(sizeof(float) * N * N);
    float* B = (float*)malloc(sizeof(float) * N * N);
    float* C = (float*)malloc(sizeof(float) * N * N);
    for(int i = 0; i < N * N; ++i){
        A[i] = dist(gen);
        B[i] = dist(gen);
    }

    // mmul
    auto startTime = std::chrono::high_resolution_clock::now();
    mmul(A, B, C, N);
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(endTime - startTime);

    std::cout << C[0] << std::endl;
    std::cout << C[N * N - 1] << std::endl;
    std::cout << duration.count() << std::endl;

    free(A);
    free(B);
    free(C);
}