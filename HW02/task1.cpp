#include <iostream>
#include <cstdlib>
#include <random>
#include <chrono>
#include "scan.h"

int main(int argc, char* argv[]){
    int N = std::atoi(argv[1]);
    float* arrInput = (float*)malloc(N * sizeof(float));
    float* arrOutput = (float*)malloc(N * sizeof(float));

    // set up random generator
    std::random_device rd; 
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0, 1.0);
    
    // fill the input array
    for(int i = 0; i < N; ++i){
        arrInput[i] = dist(gen);
        //std::cout << arrInput[i] << std::endl;
    }

    // call scan for core computations
    auto startTime = std::chrono::high_resolution_clock::now();
    scan(arrInput, arrOutput, N);
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(endTime - startTime);
    
    // print outputs
    std::cout << duration.count() << std::endl;
    std::cout << arrOutput[0] << std::endl;
    std::cout << arrOutput[N - 1] << std::endl;

    free(arrInput);
    free(arrOutput);
    return 0;
}