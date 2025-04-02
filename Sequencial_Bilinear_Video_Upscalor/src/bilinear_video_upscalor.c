#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Function to perform bilinear interpolation
void bilinear_interpolation(unsigned char *data, uint32_t input_width,
                            uint32_t input_height, uint32_t output_width,
                            uint32_t output_height, unsigned char *output, int channels)
{
    float x_ratio = (float)(input_width - 1) / (float)(output_width - 1);
    float y_ratio = (float)(input_height - 1) / (float)(output_height - 1);

    for (int i = 0; i < output_height; i++)
    {
        for (int j = 0; j < output_width; j++)
        {
            float x = x_ratio * j;
            float y = y_ratio * i;

            int x1 = (int)x;
            int y1 = (int)y;
            int x2 = (x1 + 1 < input_width) ? x1 + 1 : x1;  // Ensure bounds
            int y2 = (y1 + 1 < input_height) ? y1 + 1 : y1; // Ensure bounds

            float x_weight = x - x1;
            float y_weight = y - y1;

            for (int c = 0; c < channels; c++)
            {
                float a = data[(y1 * input_width + x1) * channels + c];
                float b = data[(y1 * input_width + x2) * channels + c];
                float c1 = data[(y2 * input_width + x1) * channels + c];
                float d = data[(y2 * input_width + x2) * channels + c];

                float pixel = a * (1.0 - x_weight) * (1.0 - y_weight) +
                              b * x_weight * (1.0 - y_weight) +
                              c1 * y_weight * (1.0 - x_weight) +
                              d * x_weight * y_weight;

                // Ensure the pixel values are within the [0, 255] range
                output[(i * output_width + j) * channels + c] = (unsigned char)fminf(255.0, fmaxf(0.0, pixel));
            }
        }
    }
}

// Function to load an image, upscale it, and save it
int upscale_image(const char *input_filename, const char *output_filename, float scale_factor)
{
    int width, height, channels;

    // Load the image
    unsigned char *image = stbi_load(input_filename, &width, &height, &channels, 0);
    if (image == NULL)
    {
        printf("Error loading image\n");
        return -1;
    }

    // Calculate new dimensions for the upscaled image
    uint32_t output_width = (uint32_t)(width * scale_factor);
    uint32_t output_height = (uint32_t)(height * scale_factor);

    // Allocate memory for the upscaled image
    unsigned char *output_image = (unsigned char *)malloc(output_width * output_height * channels);
    if (!output_image)
    {
        printf("Memory allocation failed\n");
        stbi_image_free(image);
        return -1;
    }

    // Perform bilinear interpolation to upscale the image
    bilinear_interpolation(image, width, height, output_width, output_height, output_image, channels);

    // Save the upscaled image
    if (channels == 4)
    {
        // Save as PNG with alpha (RGBA)
        if (!stbi_write_png(output_filename, output_width, output_height, channels, output_image, output_width * channels))
        {
            printf("Error saving image to %s\n", output_filename);
            free(output_image);
            stbi_image_free(image);
            return -1;
        }
    }
    else
    {
        // Save as PNG without alpha (RGB)
        unsigned char *rgb_output = (unsigned char *)malloc(output_width * output_height * 3);
        for (int i = 0; i < output_height; i++)
        {
            for (int j = 0; j < output_width; j++)
            {
                int idx = (i * output_width + j) * channels;
                int out_idx = (i * output_width + j) * 3;

                // Copy RGB channels from RGBA (ignore alpha channel)
                rgb_output[out_idx] = output_image[idx];         // R
                rgb_output[out_idx + 1] = output_image[idx + 1]; // G
                rgb_output[out_idx + 2] = output_image[idx + 2]; // B
            }
        }

        if (!stbi_write_png(output_filename, output_width, output_height, 3, rgb_output, output_width * 3))
        {
            printf("Error saving image to %s\n", output_filename);
            free(rgb_output);
            free(output_image);
            stbi_image_free(image);
            return -1;
        }

        free(rgb_output);
    }

    // Clean up and free memory
    free(output_image);
    stbi_image_free(image);

    printf("Image saved to %s\n", output_filename);
    return 0;
}

// Main function to take input arguments and call the upscale function
int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        printf("Usage: %s <input_image> <output_image> <scale_factor>\n", argv[0]);
        return -1;
    }

    const char *input_filename = argv[1];
    const char *output_filename = argv[2];
    float scale_factor = atof(argv[3]);

    // Call the upscale function
    return upscale_image(input_filename, output_filename, scale_factor);
}