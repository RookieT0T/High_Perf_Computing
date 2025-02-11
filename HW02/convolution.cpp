#include <iostream>
#include <cstdlib>

// Computes the result of applying a mask to an image as in the convolution process described in HW02.pdf.
// image is an nxn grid stored in row-major order.
// mask is an mxm grid stored in row-major order.
// Stores the result in output, which is an nxn grid stored in row-major order.
void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m){
    for(std::size_t i_out = 0; i_out < n; ++i_out){
        for(std::size_t j_out = 0; j_out < n; ++j_out){
            float temp = 0;
            for(std::size_t i_mask = 0; i_mask < m; ++i_mask){
                for(std::size_t j_mask = 0; j_mask < m; ++j_mask){
                    std::size_t x_image = i_out + i_mask - (m - 1) / 2;
                    std::size_t y_image = j_out + j_mask - (m - 1) / 2;
                    
                    bool x = (x_image >= 0) && (x_image < n);
                    bool y = (y_image >= 0) && (y_image < n);

                    if(x && y){
                        temp += mask[i_mask * m + j_mask] * image[x_image * n + y_image];
                    }
                    else if((!x) && (!y)){
                        temp += mask[i_mask * m + j_mask] * 0;
                    }
                    else{
                        temp += mask[i_mask * m + j_mask];
                    }
                }
            }
            output[i_out * n + j_out] = temp;
        }
    }
}