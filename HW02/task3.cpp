#include <iostream>
#include <cstdlib>
#include <random>
#include <chrono>
#include "matmul.h"

/*
void printArr(const double* arr, size_t n){
    std::cout << "\nstart" << std::endl;
     for(int i = 0; i < n; ++i){
        for(int k = 0; k < n; ++k){
            std::cout << arr[i * n + k] << std::endl;
        }
    }
    std::cout << "end\n" << std::endl;
}

void printVector(const std::vector<double>arr, size_t n){
    std::cout << "\nstart" << std::endl;
    for(int i = 0; i < n; ++i){
        for(int k = 0; k < n; ++k){
            std::cout << arr[i * n + k] << std::endl;
        }
    }
    std::cout << "end\n" << std::endl;
}
*/

int main(int argc, char* argv[]){
    int row = 1024;
    std::cout << row << std::endl;

    // set up random generator
    std::random_device rd; 
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-10, 10);

    // fill the values
    double* A = (double*)malloc(sizeof(double) * row * row);
    double* B = (double*)malloc(sizeof(double) * row * row);
    double* C1 = (double*)malloc(sizeof(double) * row * row);
    double* C2 = (double*)malloc(sizeof(double) * row * row);
    double* C3 = (double*)malloc(sizeof(double) * row * row);
    double* C4 = (double*)malloc(sizeof(double) * row * row);
    std::vector<double> vectorA(row * row, 0);
    std::vector<double> vectorB(row * row, 0);

    for(int i = 0; i < row * row; ++i){
        A[i] = dist(gen);
        B[i] = dist(gen);
        vectorA[i] = A[i];
        vectorB[i] = B[i];
    }

    /*
    std::cout << "Array A B" << std::endl;
    printArr(A, row);
    printArr(B, row);
    std::cout << "Vector A B" << std::endl;
    printVector(vectorA, row);
    printVector(vectorB, row);
    */

    // mmul1
    //printArr(C1, row);
    auto startTime1 = std::chrono::high_resolution_clock::now();
    mmul1(A, B, C1, row);
    auto endTime1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(endTime1 - startTime1);
    std::cout << duration1.count() << std::endl;
    std::cout << C1[row * row - 1] << std::endl;
    //printArr(C1, row);


    // mmul2
    //printArr(C2, row);
    auto startTime2 = std::chrono::high_resolution_clock::now();
    mmul2(A, B, C2, row);
    auto endTime2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(endTime2 - startTime2);
    std::cout << duration2.count() << std::endl;
    std::cout << C2[row * row - 1] << std::endl;
    //printArr(C2, row);

    // mmul3
    //printArr(C3, row);
    auto startTime3 = std::chrono::high_resolution_clock::now();
    mmul3(A, B, C3, row);
    auto endTime3 = std::chrono::high_resolution_clock::now();
    auto duration3 = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(endTime3 - startTime3);
    std::cout << duration3.count() << std::endl;
    std::cout << C3[row * row - 1] << std::endl;
    //printArr(C3, row);

    // mmul4
    //printArr(C4, row);
    auto startTime4 = std::chrono::high_resolution_clock::now();
    mmul4(vectorA, vectorB, C4, row);
    auto endTime4 = std::chrono::high_resolution_clock::now();
    auto duration4 = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(endTime4 - startTime4);
    std::cout << duration4.count() << std::endl;
    std::cout << C4[row * row - 1] << std::endl;
    //printArr(C4, row);

    free(A);
    free(B);
    free(C1);
    free(C2);
    free(C3);
    free(C4);
}