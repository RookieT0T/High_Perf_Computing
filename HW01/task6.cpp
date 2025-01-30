#include <iostream>
#include <cstdlib>

int main(int argc, char* argv[]){
    int N = std::atoi(argv[1]);

    // 0 - N with printf
    for(int i = 0; i <= N; ++i){
        if(i == N){
            printf("%d\n", i);
        }
        else{
            printf("%d ", i);
        }
    }

    // N - 0 with std::cout
    for(int i = N; i >= 0; --i){
        if(i == 0){
            std::cout << i << std::endl;
        }
        else{
            std::cout << i << " ";
        }
    }

    return 0;
}