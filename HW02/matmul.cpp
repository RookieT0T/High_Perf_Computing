#include <iostream>
#include <cstdlib>
#include <vector>

void mmul1(const double* A, const double* B, double* C, const unsigned int n){
    // row
    for(std::size_t i = 0; i < n; ++i){
        // column
        for(std::size_t j = 0; j < n; ++j){
            // index
            for(std::size_t k = 0; k < n; ++k){
                C[i * n + j] += A[i * n + k] * B[j + k * n];
            }
        }
    }
}

void mmul2(const double* A, const double* B, double* C, const unsigned int n){
    // row
    for(std::size_t i = 0; i < n; ++i){
        // index
        for(std::size_t k = 0; k < n; ++k){
            // column
            for(std::size_t j = 0; j < n; ++j){
                C[i * n + j] += A[i * n + k] * B[j + k * n];
            }
        }
    }
}

void mmul3(const double* A, const double* B, double* C, const unsigned int n){
    // column
    for(std::size_t j = 0; j < n; ++j){
        // index
        for(std::size_t k = 0; k < n; ++k){
            // row
            for(std::size_t i = 0; i < n; ++i){
                C[i * n + j] += A[i * n + k] * B[j + k * n];
            }
        }
    }
}

void mmul4(const std::vector<double>& A, const std::vector<double>& B, double* C, const unsigned int n){
    // row
    for(std::size_t i = 0; i < n; ++i){
        // column
        for(std::size_t j = 0; j < n; ++j){
            // index
            for(std::size_t k = 0; k < n; ++k){
                C[i * n + j] += A[i * n + k] * B[j + k * n];
            }
        }
    }
}