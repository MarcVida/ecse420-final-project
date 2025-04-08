#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "bicubic.h"


static float cubic(float x, float a)
{
    float absx = fabsf(x);
    if(absx < 1.0f) {
        return (a + 2.0f)*absx*absx*absx - (a + 3.0f)*absx*absx + 1.0f;
    } else if(absx < 2.0f) {
        return a*absx*absx*absx - 5.0f*a*absx*absx + 8.0f*a*absx - 4.0f*a;
    } else {
        return 0.0f;
    }
}

static inline void copyPixel(unsigned char* output, int outIndex,
                             const unsigned char* input, int inIndex)
{
    output[outIndex + 0] = input[inIndex + 0];
    output[outIndex + 1] = input[inIndex + 1];
    output[outIndex + 2] = input[inIndex + 2];
    output[outIndex + 3] = 255;
}

unsigned char* bicubic_interpolate(const unsigned char* input,
                                   unsigned in_width,
                                   unsigned in_height,
                                   unsigned *out_width,
                                   unsigned *out_height)
{

     
    *out_width  = in_width * 2;
    *out_height = in_height * 2;

    unsigned char* output = (unsigned char*)malloc((*out_width) * (*out_height) * 4);
    if(!output) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    float a_param = -0.5f;

    for(unsigned y = 0; y < *out_height; y++)
    {
        for(unsigned x = 0; x < *out_width; x++)
        {
            int outIndex = (y * (*out_width) + x) * 4;


            if((x % 2 == 0) && (y % 2 == 1))
            {

                int inX = x / 2;
                int inY = y / 2;
                int inIndex = (inY * in_width + inX) * 4;
                copyPixel(output, outIndex, input, inIndex);
            }
            else
            {

                if(x < 3 || y < 3 ||
                   x > (2*in_width - 3) ||
                   y > (2*in_height - 3))
                {

                    int inX = x / 2;
                    int inY = y / 2;
                    int inIndex = (inY * in_width + inX) * 4;
                    copyPixel(output, outIndex, input, inIndex);
                }
                else
                {

                    float input_x = x / 2.0f;
                    float input_y = y / 2.0f;

                    float r = 0.0f, g = 0.0f, b = 0.0f;

                    for(int i = -1; i < 3; i++)
                    {
                        for(int j = -1; j < 3; j++)
                        {
                            int px = (int)floorf(input_x) + i;
                            int py = (int)floorf(input_y) + j;

                            float x_dist = fabsf(px - input_x);
                            float y_dist = fabsf(py - input_y);


                            float w_x = cubic(x_dist, a_param);
                            float w_y = cubic(y_dist, a_param);

                            float w = w_x * w_y;


                            if(px < 0) px = 0;
                            if(px >= (int)in_width)  px = in_width - 1;
                            if(py < 0) py = 0;
                            if(py >= (int)in_height) py = in_height - 1;

                            int inIdx = (py * in_width + px) * 4;
                            r += input[inIdx + 0] * w;
                            g += input[inIdx + 1] * w;
                            b += input[inIdx + 2] * w;
                        }
                    }

                    output[outIndex + 0] = (unsigned char)fminf(fmaxf(r, 0.0f), 255.0f);
                    output[outIndex + 1] = (unsigned char)fminf(fmaxf(g, 0.0f), 255.0f);
                    output[outIndex + 2] = (unsigned char)fminf(fmaxf(b, 0.0f), 255.0f);
                    output[outIndex + 3] = 255;
                }
            }
        }
    }

    return output;
}
