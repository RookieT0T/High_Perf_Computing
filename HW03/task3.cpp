#include <iostream>
#include <cstdlib>
#include <random>
#include <chrono>
#include <omp.h>
#include <algorithm>
#include "msort.h"

int main(int argc, char* argv[]){
    int n = std::atoi(argv[1]);
    int t = std::atoi(argv[2]);
    int ts = std::atoi(argv[3]);

    int* arr = (int*)malloc(sizeof(int) * n);

    omp_set_num_threads(t);

    // set up random generator
    std::random_device rd; 
    std::mt19937 gen(404);
    std::uniform_int_distribution<int> dist(-1000, 1000);

    // fill the arrays
    for(int i = 0; i < n; ++i){
        arr[i] = dist(gen);
    }

    // call convolution
    auto startTime = std::chrono::high_resolution_clock::now();
    msort(arr, n, ts);
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(endTime - startTime);
    
    // print results
    std::cout << arr[0] << std::endl;
    std::cout << arr[n - 1] << std::endl;
    std::cout << duration.count() << std::endl;

    free(arr);
    return 0;
}