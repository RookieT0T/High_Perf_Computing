#include <iostream>
#include <cstdlib>
#include <random>
#include <chrono>
#include "convolution.h"

int main(int argc, char* argv[]){
    int n = std::atoi(argv[1]);
    int m = std::atoi(argv[2]);

    float* image = (float*)malloc(sizeof(float) * n * n);
    float* out = (float*)malloc(sizeof(float) * n * n);
    float* mask = (float*)malloc(sizeof(float) * m * m);

    // set up random generator
    std::random_device rd; 
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distImage(-10.0, 10.0);
    std::uniform_real_distribution<float> distMask(-1.0, 1.0);

    // fill the arrays
    //std::cout << "S\n";
    for(int i = 0; i < n * n; ++i){
        image[i] = distImage(gen);
        //std::cout << image[i] << std::endl;
    }
    //std::cout << "E\n";
    for(int i = 0; i < m * m; ++i){
        mask[i] = distMask(gen);
        //std::cout << mask[i] << std::endl;
    }

    // call convolution
    auto startTime = std::chrono::high_resolution_clock::now();
    convolve(image, out, n, mask, m);
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(endTime - startTime);

    /*
    for(int i = 0; i < n * n; ++i){
        std::cout << out[i] << std::endl;
    }
    */
    
    // print results
    std::cout << duration.count() << std::endl;
    std::cout << out[0] << std::endl;
    std::cout << out[n * n - 1] << std::endl;

    free(image);
    free(out);
    free(mask);
    return 0;
}