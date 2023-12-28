#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <filesystem>
#include <vector>
#include <stdio.h>
#define CHANNELS 3
#define BLUR_SIZE 3
using namespace cv;
__global__
void blurKernel(unsigned char* in, unsigned char* out, int w, int h) {
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    if (Col < w && Row < h) {
        int pixValR = 0, pixValG = 0, pixValB = 0;
        int pixels = 0;
        // Get the avg of the surrounding
        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
                int curRow = Row + blurRow;
                int curCol = Col + blurCol;
                // Verify if we have a valid img pixel
                if (curRow > -1 && curRow < h && curCol > -1 && curCol < w) {
                    pixValB += in[(curRow * w + curCol) * 3];
                    pixValG += in[(curRow * w + curCol) * 3 + 1];
                    pixValR += in[(curRow * w + curCol) * 3 + 2];
                    pixels++;
                }
            }
        }
        // Write our new pixel value out
        out[(Row * w + Col) * 3] = (unsigned char)(pixValB / pixels);
        out[(Row * w + Col) * 3 + 1] = (unsigned char)(pixValG / pixels);
        out[(Row * w + Col) * 3 + 2] = (unsigned char)(pixValR / pixels);
    }
}
__global__
void colorToGreyscaleConversion(unsigned char* Pout, unsigned char* Pin, int width, int height) {
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    if (Col < width && Row < height) {
        // get 1D coordinate for the grayscale image
        int greyOffset = Row * width + Col;
        // one can think of the RGB image having
        // CHANNEL times columns than the grayscale image
        int rgbOffset = greyOffset * CHANNELS;
        unsigned char r = Pin[rgbOffset]; // red value for pixel
        unsigned char g = Pin[rgbOffset + 2]; // green value for pixel
        unsigned char b = Pin[rgbOffset + 3]; // blue value for pixel
        // perform the rescaling and store it
        // We multiply by floating point constants
        Pout[greyOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}
void grayScale()
{
    
    std::string pathName="D:\\fmatt\\Documents\\University\\UNSA\\Semestre VIII\\Parelallel and Distributed Computing\\Laboratory\\Laboratory 06\\Laboratory 06\\";
    std::string filename= "dog_img.jpg";
    // std::string filename= "person.jpg";
    // std::string filename= "tiger.jpg";
    std::string imagePathStr = pathName+filename;

    cv::Mat img = cv::imread(imagePathStr);
    int width = img.cols,height = img.rows;

    // Allocate memory for input and output images on the GPU
    unsigned char* d_Pin, * d_Pout;
    cudaMalloc((void**)&d_Pin, width * height * CHANNELS * sizeof(unsigned char));
    cudaMalloc((void**)&d_Pout, width * height * sizeof(unsigned char));

    // Copy input image data to the GPU
    cudaMemcpy(d_Pin, img.data, width * height * CHANNELS * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch the CUDA kernel
    colorToGreyscaleConversion << < gridSize, blockSize >> > (d_Pout, d_Pin, width, height);

    // Copy the result back to the host
    unsigned char* h_Pout = new unsigned char[width * height];
    cudaMemcpy(h_Pout, d_Pout, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Create an OpenCV Mat for the output grayscale image
    cv::Mat outputImage(height, width, CV_8UC1, h_Pout);

    // Display the images using OpenCV (optional)
    cv::imshow("Original Image", img);
    cv::imshow("Grayscale Image", outputImage);
    cv::waitKey(0);
    cv::destroyAllWindows();
    // Clean up
    delete[] h_Pout;
    cudaFree(d_Pin);
    cudaFree(d_Pout);
}
int main()
{
    std::string pathName="D:\\fmatt\\Documents\\University\\UNSA\\Semestre VIII\\Parelallel and Distributed Computing\\Laboratory\\Laboratory 06\\Laboratory 06\\";
    std::string filename= "dog_img.jpg";
    // std::string filename= "person.jpg";
    // std::string filename= "tiger.jpg";
    std::string imagePathStr = pathName+filename;

    cv::Mat img = cv::imread(imagePathStr);
    int width = img.cols;
    int height = img.rows;
    cv::Mat outputImage(height, width, CV_8UC3);

    unsigned char* d_input, * d_output;
    cudaMalloc((void**)&d_input, width * height * 3 * sizeof(unsigned char));
    cudaMalloc((void**)&d_output, width * height * 3 * sizeof(unsigned char));

    cudaMemcpy(d_input, img.data, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    blurKernel << <gridSize, blockSize >> > (d_input, d_output, width, height);

    cudaMemcpy(outputImage.data, d_output, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    cv::imshow("Input Image", img);
    cv::imshow("Blurred Image", outputImage);
    cv::waitKey(0);

    return 0;
}
