
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"
#include "gputimer.h"
#include <stdio.h>


cudaError_t bicubicInterpCuda(unsigned char* input, unsigned char* output, int inputWidth, int inputHeight, int numThreads);

__global__ void upscale(unsigned char* input, unsigned char* output, int numPixelsPerThread, int inputWidth, int inputHeight){
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int startingPixel = threadId * numPixelsPerThread;
    
    for(int currPixel = startingPixel; currPixel < startingPixel + numPixelsPerThread; currPixel++){
        int x = currPixel % (inputWidth * 2);
        int y = currPixel / (inputWidth * 2);
        if((currPixel % (inputWidth*2)) % 2 == 0 && (currPixel / (inputWidth*2)) % 2 == 1){
            // Take pixel value from original image
            for(int c=0; c<4; c++){
                output[currPixel*4 + c] = input[4*(x/2+y/2*inputWidth) + c];
            }
            output[currPixel*4+3] = 255;
        }else{
            // Handle edge cases where 4x4 kernel wont fit
            if(x < 3 || y < 3 || x > (inputWidth*2 - 3) || y > (inputHeight*2 - 3)){
                // (temp) take a pixel value from original
                for(int c=0; c<4; c++){
                    output[4*currPixel+c] = input[4*(x/2+y/2*inputWidth)+c];
                }
                output[4*currPixel+3] = 255;
                
            }else{

                float input_x = x / 2.0f;
                float input_y = y / 2.0f;
                float x_dist;
                float y_dist;
                float a_param = -0.5f;
                float x_weight = 0.0f;
                float y_weight = 0.0f;
                
                float r = 0, g=0, b=0, total_weight=0;
                for(int i = -1; i < 3; i++){
                    for(int j = -1; j < 3; j++){
                        int px = floor(input_x) + i;
                        int py = floor(input_y) + j;
                        x_dist = abs(px - input_x);
                        y_dist = abs(py - input_y);
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
                    }
                }
                output[4*currPixel] = r;
                output[4*currPixel + 1] = g;
                output[4*currPixel + 2] = b;
                output[4*currPixel + 3] = 255;

            }
        }
    }
}


int main(int argc, char* argv[])
{
    if (argc != 4) {
        printf("Proper usage: ./bicubic_cuda <name of input png> <name of output png> <# threads>");
        return 1;
    }
    const int NUM_THREADS = atoi(argv[3]);
    if (NUM_THREADS == 0) {
        printf("# threads must be valid non-zero integer");
        return 1;
    }

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
    cudaError_t cudaStatus = bicubicInterpCuda(image, new_image, inputWidth, inputHeight, NUM_THREADS);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
    
    
    // Save output image
    error = lodepng_encode32_file(argv[2], new_image, inputWidth * 2, inputHeight * 2);
    if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
    
    // Free memory
    free(image);
    free(new_image);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
cudaError_t bicubicInterpCuda(unsigned char* input, unsigned char* output, int inputWidth, int inputHeight, int numThreads){
    cudaError_t cudaStatus;
    unsigned char* dev_input = 0;
    unsigned char* dev_output = 0;
    GpuTimer timer;


    printf("%d, %d, %d\n", inputWidth, inputHeight, numThreads);

    int numBlocks = ceil(((float)numThreads)/1024); 
    int numPixelsPerThread = ceil((inputWidth*2)*(inputHeight*2)/((float)numThreads));
    int numActualThreads = numThreads % 32;

    printf("Num blocks %d, num pixels per thread %d\n", numBlocks, numPixelsPerThread);

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_input, inputHeight * inputWidth * 4 * sizeof(char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_output, (inputHeight) * (inputWidth) * 4 * 4 * sizeof(char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_input, input, inputHeight * inputWidth * 4 * sizeof(char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_output, output, inputHeight * inputWidth * 4 * 4 * sizeof(char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    timer.Start();
    upscale<<<numBlocks, numThreads>>>(dev_input, dev_output, numPixelsPerThread, inputWidth, inputHeight);
    timer.Stop();
    printf("Elapsed time: %f\n", timer.Elapsed());

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(output, dev_output, (inputHeight*2) * (inputWidth*2) * 4 * sizeof(char), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    Error:
    cudaFree(dev_input);
    cudaFree(dev_output);
    return cudaStatus;

}
