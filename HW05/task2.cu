#include <cuda.h>
#include <stdio.h>
#include <random>

__global__ void kernel(int* dA, int a){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    dA[index] = a * threadIdx.x + blockIdx.x;
}

int main(){
    // random generator
    std::mt19937 gen(404);
    std::uniform_int_distribution<int> dist(1, 5);
    int ranNum = dist(gen);

    // host array
    int* hA = (int*)malloc(16 * sizeof(int));
    // device array
    int* dA;
    cudaMalloc((void**)&dA, sizeof(int) * 16);
    cudaMemset(dA, 0, sizeof(int) * 16);

    // two blocks and eight threads per block
    kernel<<<2, 8>>>(dA, ranNum);
    cudaDeviceSynchronize();

    cudaMemcpy(hA, dA, sizeof(int) * 16, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 16; ++i) {
        printf("%d ", hA[i]);
    }
    printf("\n");

    cudaFree(dA);
    free(hA);
    return 0;
}