#include <cuda.h>
#include <stdio.h>

__global__ void kernel(){
    int index = threadIdx.x;
    int result = 1;
    for(int i = 1; i <= index + 1; ++i){
        result *= i;
    }
    printf("%d!=%d\n", index + 1, result);
}

int main(){
    // one block and eight threads
    kernel<<<1, 8>>>();
    cudaDeviceSynchronize();
    return 0;
}