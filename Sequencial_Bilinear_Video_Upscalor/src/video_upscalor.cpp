#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <iomanip>
#include <filesystem>

extern "C" {
    void upscale_image(const char *input_filename, const char *output_filename, float scale_factor);
}

std::string format_filename(const std::string& prefix, int index, const std::string& ext) {
    std::ostringstream oss;
    oss << prefix << std::setw(5) << std::setfill('0') << index << ext;
    return oss.str();
}

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("Usage: %s <input_video> <output_video> <scale_factor>\n", argv[0]);
        return -1;
    }

    const std::string input_video = argv[1];
    const std::string output_video = argv[2];
    const std::string temp_dir = "temp"; // Temp folder
    const std::string temp_prefix = "temp_frame_";
    const std::string ext = ".png";
    float scale_factor = std::stof(argv[3]);

    // Create temp folder if it doesn't exist
    if (!std::filesystem::exists(temp_dir)) {
        std::filesystem::create_directory(temp_dir);
    }

    cv::VideoCapture cap(input_video);
    if (!cap.isOpened()) {
        printf("Cannot open input video.\n");
        return -1;
    }

    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    // Extract and upscale
    int count = 0;
    cv::Mat frame;

    std::vector<cv::Mat> upscaled_frames;

    while (cap.read(frame)) {
        std::string input_img = temp_dir + "/" + format_filename(temp_prefix, count, ext);
        std::string output_img = temp_dir + "/" + format_filename(temp_prefix + "up_", count, ext);

        // Save original frame
        cv::imwrite(input_img, frame);

        // Call your C bilinear interpolator
        upscale_image(input_img.c_str(), output_img.c_str(), scale_factor);

        // Load the upscaled result back
        cv::Mat upscaled = cv::imread(output_img);
        if (upscaled.empty()) {
            printf("Failed to read upscaled frame: %s\n", output_img.c_str());
            return -1;
        }

        upscaled_frames.push_back(upscaled);
        count++;
        printf("Processed frame %d / %d\r", count, total_frames);
    }

    // Use size of the first frame to init writer
    if (upscaled_frames.empty()) {
        printf("No frames processed.\n");
        return -1;
    }

    cv::Size outSize = upscaled_frames[0].size();
    
    // Try using 'XVID' codec or another available one
    cv::VideoWriter writer(output_video, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps, outSize);
    if (!writer.isOpened()) {
        printf("Error opening video writer.\n");
        return -1;
    }

    for (const auto& f : upscaled_frames) {
        writer.write(f);
    }

    writer.release();
    cap.release();

    // Cleanup temp files and temp folder
    for (int i = 0; i < count; i++) {
        std::filesystem::remove(temp_dir + "/" + format_filename(temp_prefix, i, ext));
        std::filesystem::remove(temp_dir + "/" + format_filename(temp_prefix + "up_", i, ext));
    }

    // Remove the temp directory itself after files are deleted
    std::filesystem::remove_all(temp_dir);

    printf("\nDone. Saved to %s\n", output_video.c_str());
    return 0;
}
