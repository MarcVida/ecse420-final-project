#include "lodepng.h"
#include <stdio.h>
#include <windows.h> 
#include <math.h>

int bicubicInterpSeq(unsigned char* input, unsigned char* output, int inputWidth, int inputHeight);

void upscale(unsigned char* input, unsigned char* output, int numPixelsPerThread, int inputWidth, int inputHeight){
    printf("Need to process %d pixels\n", numPixelsPerThread);
    int verbose = 0;
    for(int currPixel = 0; currPixel < numPixelsPerThread; currPixel++){
        int x = currPixel % (inputWidth * 2);
        int y = currPixel / (inputWidth * 2);
        if(verbose)printf("X, Y: %d, %d", x,y);
        if((currPixel % (inputWidth*2)) % 2 == 0 && (currPixel / (inputWidth*2)) % 2 == 1){
            if(verbose)printf("Going in here\n");
            // Take pixel value from original image
            for(int c=0; c<4; c++){
                output[currPixel*4 + c] = input[4*(x/2+y/2*inputWidth) + c];
            }
            output[currPixel*4+3] = 255;
        }else{
            float input_x = x / 2.0f;
            float input_y = y / 2.0f;
            // Handle edge cases where 4x4 kernel wont fit
            if(x < 4 || y < 4 || x > (inputWidth*2 - 4) || y > (inputHeight*2 - 4)){
                // (temp) take a pixel value from original
                for(int c=0; c<4; c++){
                    output[4*currPixel+c] = input[4*(x/2+y/2*inputWidth)+c];
                }
                output[4*currPixel+3] = 255;
                
            }else{

                float x_dist;
                float y_dist;
                float a_param = -0.5f;
                float x_weight = 0.0f;
                float y_weight = 0.0f;
                if(verbose) printf("Going in here3: %f %f\n", input_x, input_y);
                float r = 0, g=0, b=0, total_weight=0;
                for(int i = -1; i < 3; i++){
                    for(int j = -1; j < 3; j++){
                        int px = floor(input_x) + i;
                        int py = floor(input_y) + j;
                        
                        if(px < 0){
                            px=0;
                        }else if(px >= inputWidth){
                            px = inputWidth-1;
                        }
                        if(py < 0){
                            py=0;
                        }else if(py >= inputHeight){
                            py = inputHeight-1;
                        }

                        

                        if(verbose)printf("Px: %d, Py:%d\n", px, py);
                        x_dist = fabs(px - input_x);
                        y_dist = fabs(py - input_y);
                        if(x_dist < 1.0f){
                            x_weight = (a_param + 2)*pow(x_dist,3) - (a_param + 3)*pow(x_dist, 2) + 1;
                        }else if(x_dist < 2.0f){
                            x_weight = a_param*pow(x_dist, 3) - 5*a_param*pow(x_dist,2) + 8*a_param*x_dist - 4*a_param;
                        }else{
                            x_weight = 0;
                        }

                        if(y_dist < 1.0f){
                            y_weight = (a_param + 2)*pow(y_dist,3) - (a_param + 3)*pow(y_dist, 2) + 1;
                        }else if(y_dist < 2.0f){
                            y_weight = a_param*pow(y_dist, 3) - 5*a_param*pow(y_dist,2) + 8*a_param*y_dist - 4*a_param;
                        }else{
                            y_weight = 0;
                        }
                        r += input[4*(px + py*inputWidth)]* x_weight * y_weight;
                        g += input[4*(px + py*inputWidth)+1]* x_weight * y_weight;
                        b += input[4*(px + py*inputWidth)+2]* x_weight * y_weight;
                        total_weight += x_weight * y_weight;
                    }
                }
                output[4*currPixel] = r/total_weight;
                output[4*currPixel + 1] = g/total_weight;
                output[4*currPixel + 2] = b/total_weight;
                output[4*currPixel + 3] = 255;

            }
        }
    }
    printf("Completed all loops");
}


int main(int argc, char* argv[])
{
    if (argc != 3) {
        printf("Proper usage: ./bicubic_sqeuential <name of input png> <name of output png>");
        return 1;
    }
    setvbuf(stdout,NULL, _IONBF,0);
    // Variable declarations
    unsigned error;
    unsigned char* image, * new_image;
    unsigned inputWidth, inputHeight;

    error = lodepng_decode32_file(&image, &inputWidth, &inputHeight, argv[1]);
    if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

    // Allocate output image space
    new_image = (unsigned char*)malloc(inputWidth * 2 * inputHeight * 2 * 4 * sizeof(unsigned char));
    for(int i=0; i<inputWidth*inputHeight*16; i++){
        new_image[i] = 0;
    }

    // Prep and perform cuda function
    bicubicInterpSeq(image, new_image, inputWidth, inputHeight);
    printf("Interpolation complete\n");

    // Save output image
    error = lodepng_encode32_file(argv[2], new_image, inputWidth * 2, inputHeight * 2);
    if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
    
    // Free memory
    free(image);
    free(new_image);

    return 0;
}
int bicubicInterpSeq(unsigned char* input, unsigned char* output, int inputWidth, int inputHeight){
    unsigned char* dev_input = 0;
    unsigned char* dev_output = 0;

    printf("%d, %d, %d\n", inputWidth, inputHeight, 1);
    
    LARGE_INTEGER frequency;
    LARGE_INTEGER start;
    LARGE_INTEGER end;
    double elapsed_time;
    
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    printf("Beginning Upscale\n");
    upscale(input, output, inputHeight*inputWidth*4, inputWidth, inputHeight);
    printf("Interpolaton complete 1");
    QueryPerformanceCounter(&end);
    elapsed_time = (double)(end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;
    printf("Algorithm execution time: %.3f milliseconds\n", elapsed_time);

    return 0;

}
