#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>

__global__ void calcHistKernel(unsigned char* d_channel, int* d_hist, int num_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pixels) {
        atomicAdd(&d_hist[d_channel[idx]], 1);
    }
}

void calculateCustomHistCuda(const cv::Mat& channel, int* h_hist) {
    int num_pixels = channel.rows * channel.cols;
    unsigned char* d_channel;
    int* d_hist;

    cudaMalloc(&d_channel, num_pixels * sizeof(unsigned char));
    cudaMalloc(&d_hist, 256 * sizeof(int));
    cudaMemset(d_hist, 0, 256 * sizeof(int));

    cudaMemcpy(d_channel, channel.data, num_pixels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (num_pixels + threads - 1) / threads;
    calcHistKernel<<<blocks, threads>>>(d_channel, d_hist, num_pixels);
    cudaDeviceSynchronize();

    cudaMemcpy(h_hist, d_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_channel);
    cudaFree(d_hist);
}

int main() {
    cv::Mat src = cv::imread("resim.jpg", cv::IMREAD_COLOR);
    if (src.empty()) return -1;

    std::vector<cv::Mat> channels;
    cv::split(src, channels);

    int custom_b_hist[256], custom_g_hist[256], custom_r_hist[256];
    
    calculateCustomHistCuda(channels[0], custom_b_hist);
    calculateCustomHistCuda(channels[1], custom_g_hist);
    calculateCustomHistCuda(channels[2], custom_r_hist);

    std::cout << "CUDA atomicAdd ile custom histogramlar başarıyla hesaplandı." << std::endl;
    return 0;
}
