#include <cstddef>
#include <cstdlib>
#include <omp.h>

void merge(int* arr, int startIndex, int midIndex, int endIndex){
    int leftHalfSize = midIndex - startIndex + 1;
    int rightHalfSize = endIndex - midIndex;
    int* left = new int[leftHalfSize];
    int* right = new int[rightHalfSize];

    for(int i = 0; i < leftHalfSize; ++i){
        left[i] = arr[startIndex + i];
    }
    for(int i = 0; i < rightHalfSize; ++i){
        right[i] = arr[midIndex + 1 + i];
    }

    int startPoint = startIndex;
    int leftPoint = 0;
    int rightPoint = 0;
    while(leftPoint < leftHalfSize && rightPoint < rightHalfSize){
        if(left[leftPoint] <= right[rightPoint]){
            arr[startPoint] = left[leftPoint];
            leftPoint++;
        }
        else{
            arr[startPoint] = right[rightPoint];
            rightPoint++;
        }
        startPoint++;
    }

    while(leftPoint < leftHalfSize){
        arr[startPoint++] = left[leftPoint++];
    }
    while(rightPoint < rightHalfSize){
        arr[startPoint++] = right[rightPoint++];
    }

    delete[] left;
    delete[] right;
    return;
}

void serialSort(int* arr, int startIndex, int endIndex){
    for(int i = startIndex; i <= endIndex; ++i){
        int index = i;
        for(int k = i + 1; k <= endIndex; ++k){
            if(arr[k] < arr[index]){
                index = k;
            }
        }
        int temp = arr[i];
        arr[i] = arr[index];
        arr[index] = temp;
    }
    return;
}

void parallelSort(int* arr, int startIndex, int endIndex, const std::size_t threshold){
    if(endIndex - startIndex + 1 < threshold){
        serialSort(arr, startIndex, endIndex);
        return;
    }

    int midIndex = (endIndex - startIndex) / 2 + startIndex;
    #pragma omp task
    parallelSort(arr, startIndex, midIndex, threshold);
    #pragma omp task
    parallelSort(arr, midIndex + 1, endIndex, threshold);
    #pragma omp taskwait

    merge(arr, startIndex, midIndex, endIndex);
    return;
}

// This function does a merge sort on the input array "arr" of length n. 
// You can add more functions as needed to complete the merge sort,
// but do not change this file. Declare and define your addtional
// functions in the msort.cpp file, but the calls to your addtional functions
// should be wrapped in the "msort" function.

// "threshold" is the lower limit of array size where your function would 
// start making parallel recursive calls. If the size of array goes below
// the threshold, a serial sort algorithm will be used to avoid overhead
// of task scheduling
void msort(int* arr, const std::size_t n, const std::size_t threshold){
    #pragma omp parallel
    #pragma omp single
    parallelSort(arr, 0, n - 1, threshold);
    return;
}