#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "bicubic.h"

// Cubic convolution kernel with parameter a (common choice: -0.5).
static double cubic(double x, double a)
{
    double absx = fabs(x);
    if (absx <= 1.0)
        return (a + 2) * absx * absx * absx - (a + 3) * absx * absx + 1;
    else if (absx < 2.0)
        return a * absx * absx * absx - 5 * a * absx * absx + 8 * a * absx - 4 * a;
    else
        return 0.0;
}

// Clamp integer x to the range [lower, upper]
static int clamp(int x, int lower, int upper)
{
    if (x < lower) return lower;
    if (x > upper) return upper;
    return x;
}

unsigned char* bicubic_interpolate(const unsigned char* input,
                                   unsigned in_width,
                                   unsigned in_height,
                                   float scale,
                                   unsigned *out_width,
                                   unsigned *out_height)
{
    if(scale <= 0) scale = 1.0f;
    *out_width = (unsigned)(in_width * scale);
    *out_height = (unsigned)(in_height * scale);

    unsigned char* output = (unsigned char*)malloc((*out_width) * (*out_height) * 4);
    if(!output) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }
    
    double a = -0.5; // common value for bicubic interpolation

    for (unsigned y = 0; y < *out_height; y++)
    {
        double in_y = y / scale;
        int y_int = (int)floor(in_y);
        double y_frac = in_y - y_int;
        
        for (unsigned x = 0; x < *out_width; x++)
        {
            double in_x = x / scale;
            int x_int = (int)floor(in_x);
            double x_frac = in_x - x_int;
            
            double r = 0, g = 0, b = 0, a_channel = 0;
            double weight_sum = 0.0;
            
            // Process a 4x4 neighborhood.
            for (int m = -1; m <= 2; m++)
            {
                int yy = clamp(y_int + m, 0, in_height - 1);
                double w_y = cubic(m - y_frac, a);
                for (int n = -1; n <= 2; n++)
                {
                    int xx = clamp(x_int + n, 0, in_width - 1);
                    double w_x = cubic(n - x_frac, a);
                    double weight = w_x * w_y;
                    
                    int index = (yy * in_width + xx) * 4;
                    r += weight * input[index + 0];
                    g += weight * input[index + 1];
                    b += weight * input[index + 2];
                    a_channel += weight * input[index + 3];
                    weight_sum += weight;
                }
            }
            
            int out_index = (y * (*out_width) + x) * 4;
            if(weight_sum != 0)
            {
                r /= weight_sum;
                g /= weight_sum;
                b /= weight_sum;
                a_channel /= weight_sum;
            }
            output[out_index + 0] = (unsigned char)(clamp((int)round(r), 0, 255));
            output[out_index + 1] = (unsigned char)(clamp((int)round(g), 0, 255));
            output[out_index + 2] = (unsigned char)(clamp((int)round(b), 0, 255));
            output[out_index + 3] = (unsigned char)(clamp((int)round(a_channel), 0, 255));
        }
    }
    
    return output;
}
