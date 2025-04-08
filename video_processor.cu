#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>

#include "lodepng.h"      
#include "bicubic.h"

#ifdef _WIN32
    #include <direct.h>
    #include <errno.h>
    int create_directory(const char* dirname) {
        int ret = _mkdir(dirname);
        if(ret != 0 && errno != EEXIST) { 
            fprintf(stderr, "Failed to create directory: %s (errno: %d)\n", dirname, errno);
            return ret;
        }
        return 0;
    }
#else
    #include <sys/stat.h>
    #include <errno.h>
    int create_directory(const char* dirname) {
        int ret = mkdir(dirname, 0777);
        if(ret != 0 && errno != EEXIST) { 
            fprintf(stderr, "Failed to create directory: %s (errno: %d)\n", dirname, errno);
            return ret;
        }
        return 0;
    }
#endif


int run_command(const char* cmd) {
    int ret = system(cmd);
    if(ret != 0) {
        fprintf(stderr, "Command failed: %s\n", cmd);
    }
    return ret;
}

int file_exists(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if(fp) {
        fclose(fp);
        return 1;
    }
    return 0;
}

int main(int argc, char* argv[])
{
    if(argc < 3) {
        printf("Usage: %s input_video output_video\n", argv[0]);
        return 1;
    }
    
    const char* input_video = argv[1];
    const char* output_video = argv[2];

    
    const char* extract_dir = "extracted_frames";
    const char* upscale_dir = "upscaled_frames";
    
    if(create_directory(extract_dir) != 0) {
        fprintf(stderr, "Failed to create directory: %s\n", extract_dir);
        return 1;
    }
    if(create_directory(upscale_dir) != 0) {
        fprintf(stderr, "Failed to create directory: %s\n", upscale_dir);
        return 1;
    }
    
    char extract_cmd[512];
    snprintf(extract_cmd, sizeof(extract_cmd),
             "ffmpeg -i \"%s\" -qscale:v 2 \"%s/frame_%%04d.png\"",
             input_video, extract_dir);
    printf("Extracting frames...\n");
    if(run_command(extract_cmd) != 0) {
        fprintf(stderr, "Frame extraction failed. Ensure FFmpeg is installed.\n");
        return 1;
    }
    
    int frame_index = 1;
    char in_frame[256], out_frame[256];
    unsigned char *image = NULL, *upscaled = NULL;
    unsigned width, height, new_width, new_height;
    unsigned error;
    
    printf("Processing frames...\n");
    while(1) {
        snprintf(in_frame, sizeof(in_frame), "%s/frame_%04d.png", extract_dir, frame_index);
        if(!file_exists(in_frame)) {
            break;
        }
        
        error = lodepng_decode32_file(&image, &width, &height, in_frame);
        if(error) {
            fprintf(stderr, "Error loading frame %s: %u: %s\n", in_frame, error, lodepng_error_text(error));
            free(image);
            break;
        }
        
        upscaled = bicubic_interpolate(image, width, height,&new_width, &new_height);
        if(!upscaled) {
            fprintf(stderr, "Error upscaling frame %s\n", in_frame);
            free(image);
            break;
        }
        
        snprintf(out_frame, sizeof(out_frame), "%s/frame_%04d.png", upscale_dir, frame_index);
        error = lodepng_encode32_file(out_frame, upscaled, new_width, new_height);
        if(error) {
            fprintf(stderr, "Error saving upscaled frame %s: %u: %s\n", out_frame, error, lodepng_error_text(error));
            free(image);
            free(upscaled);
            break;
        }
        
        printf("Processed frame %04d: %s -> %s\n", frame_index, in_frame, out_frame);
        
        free(image);
        free(upscaled);
        image = NULL;
        upscaled = NULL;
        frame_index++;
    }
    
    if(frame_index == 1) {
        fprintf(stderr, "No frames processed. Exiting.\n");
        return 1;
    }
    
    char assemble_cmd[512];
    snprintf(assemble_cmd, sizeof(assemble_cmd),
             "ffmpeg -framerate 25 -i \"%s/frame_%%04d.png\" -c:v libx264 -pix_fmt yuv420p \"%s\"",
             upscale_dir, output_video);
    printf("Assembling video...\n");
    if(run_command(assemble_cmd) != 0) {
        fprintf(stderr, "Video assembly failed.\n");
        return 1;
    }
    
    printf("Video processing complete. Output saved to: %s\n", output_video);
    
    
    return 0;
}
