#include <cstdlib>
#include <cstddef>
#include <omp.h>

void mmul(const float* A, const float* B, float* C, const std::size_t n){
    // row
    #pragma omp parallel for
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