#include <stdio.h>
#include <stdlib.h>
#include "lodepng.h"
#include <cuda_runtime.h>             // Handles all the CUDA syntax
#include <device_launch_parameters.h> // Handles device parameters
#include "lodepng.h"
#include "gputimer.h"

/**
 * @brief Checks if a CUDA call failed. If so, prints the given error message and jumps to goToLabel.
 */
#define checkCudaError(cudaStatus, errorMessage, goToLabel)                     \
  if (cudaStatus != cudaSuccess)                                                \
  {                                                                             \
    fprintf(stderr, "[%s] %s\n", errorMessage, cudaGetErrorString(cudaStatus)); \
    goto goToLabel;                                                             \
  }

/**
 * @brief Performs bilinear interpolation on an image on a set of pixels determined by the thread ID.
 * 
 * @param image - The original image
 * @param out_image - The output image
 * @param width - The width of the original image
 * @param height - The height of the original image
 * @param out_width - The width of the output image
 * @param out_height - The height of the output image
 * @param factorX - The horizontal scaling factor
 * @param factorY - The vertical scaling factor
 * @param pixelsPerThread - The number of pixels to process
 * @return void
 */
__global__ void interpolate_kernel(
    unsigned char* image, 
    unsigned char* out_image, 
    unsigned width, 
    unsigned height, 
    unsigned out_width, 
    unsigned out_height, 
    float factorX,
    float factorY,
    int pixelsPerThread) 
{
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x; // Unique thread ID
    unsigned start = tid * pixelsPerThread;
    unsigned end = min(start + pixelsPerThread, out_width * out_height);

    for (unsigned idx = start; idx < end; idx++) {
        // Get the output coordinates
        int i = idx / out_width;
        int j = idx % out_width;

        // Scale the output coordinates to the original image
        float y = i / factorY;
        float x = j / factorX;

        // Get the nearest integer coordinates in the original image
        int y0 = floor(y);
        int x0 = floor(x);
        int y1 = ceil(y);
        int x1 = ceil(x);

        // Get the distance between the scaled output coordinates and the nearest integer coordinates
        float dy = y - y0;
        float dx = x - x0;

        // Interpolate
        for (int ch = 0; ch < 4; ch++) {
            float p0 = ((1-dx) * image[4 * (y0 * width + x0) + ch] + dx * image[4 * (y0 * width + x1) + ch]);
            float p1 = ((1-dx) * image[4 * (y1 * width + x0) + ch] + dx * image[4 * (y1 * width + x1) + ch]);
            out_image[4 * (i * out_width + j) + ch] = (unsigned char)((1 - dy) * p0 + dy * p1);
        }
    }
}

int main(int argc, char* argv[]) {
    // Parse arguments
    if (argc != 6) {
        printf("Usage: %s <#_blocks> <#_threads_per_block> <upscale_factor> <input_file> <output_file>\n", argv[0]);
        return 1;
    }
    int num_blocks = atoi(argv[1]);
    if (num_blocks <= 0) {
        printf("Number of blocks must be greater than 0\n");
        return 2;
    }
    int num_threads_per_block = atoi(argv[2]);
    if (num_threads_per_block <= 0) {
        printf("Number of threads per block must be greater than 0\n");
        return 2;
    }
    int factor = atoi(argv[3]);
    if (factor <= 0) {
        printf("Upscale factor must be greater than 0\n");
        return 2;
    }
    char* input_filename = argv[4];
    char* output_filename = argv[5];

    GpuTimer timer;

    // Load image
    unsigned char* image = NULL;
    unsigned width, height;
    unsigned error;
    error = lodepng_decode32_file(&image, &width, &height, input_filename);
    if (error) {
        printf("error %u: %s\n", error, lodepng_error_text(error));
        return 3;
    }

    // Allocate host memory for the output image
    unsigned out_width = width * factor;
    unsigned out_height = height * factor;
    unsigned char* out_image = (unsigned char*)malloc(out_width * out_height * 4);
    
    // Allocate device memory and copy image to device
    unsigned char* d_image = NULL;
    unsigned char* d_out_image = NULL;
    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc(&d_image, width * height * 4 * sizeof(unsigned char));
    checkCudaError(cudaStatus, "Failed to allocate device memory", Error);
    cudaStatus = cudaMemcpy(d_image, image, width * height * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    checkCudaError(cudaStatus, "Failed to copy image to device memory", Error);
    cudaStatus = cudaMalloc(&d_out_image, out_width * out_height * 4 * sizeof(unsigned char));
    checkCudaError(cudaStatus, "Failed to allocate device memory", Error);
    
    // Compute remaining kernel parameters
    float factorX = (float)(factor * width - 1) / (float)(width - 1);
    float factorY = (float)(factor * height - 1) / (float)(height - 1);
    unsigned pixelsPerThread = (unsigned)ceil((float)out_width * out_height / (num_blocks * num_threads_per_block));

    // Launch kernel
    timer.Start();
    interpolate_kernel<<<num_blocks, num_threads_per_block>>>(d_image, d_out_image, width, height, out_width, out_height, factorX, factorY, pixelsPerThread);
    cudaStatus = cudaGetLastError();
    checkCudaError(cudaStatus, "Failed to launch kernel", Error);
    cudaStatus = cudaDeviceSynchronize();
    checkCudaError(cudaStatus, "Failed to synchronize", Error);
    timer.Stop();
    printf("Elapsed time: %f ms\n", timer.Elapsed());

    // Copy output image back to host
    cudaStatus = cudaMemcpy(out_image, d_out_image, out_width * out_height * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    checkCudaError(cudaStatus, "Failed to copy image to host memory", Error);

    // Save output image
    error = lodepng_encode32_file(output_filename, out_image, out_width, out_height);
    if (error) {
        printf("error %u: %s\n", error, lodepng_error_text(error));
        return 4;
    }
    printf("Output image written to %s\n", output_filename);

Error:
    // Free allocated memory
    cudaFree(d_image);
    cudaFree(d_out_image);
    free(image);
    free(out_image);
    return 0;
}