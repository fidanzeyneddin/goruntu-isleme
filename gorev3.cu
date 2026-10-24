#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>

__global__ void bgrToGrayKernel(unsigned char* d_in, unsigned char* d_out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h) {
        int c_idx = (y * w + x) * 3;
        int g_idx = y * w + x;
        unsigned char b = d_in[c_idx];
        unsigned char g = d_in[c_idx + 1];
        unsigned char r = d_in[c_idx + 2];
        d_out[g_idx] = (r * 4899 + g * 9617 + b * 1868 + 16384) >> 15;
    }
}

int main() {
    cv::Mat src = cv::imread("resim.jpg", cv::IMREAD_COLOR);
    if (src.empty()) return -1;

    int w = src.cols; int h = src.rows;
    unsigned char *d_in, *d_out;
    cudaMalloc(&d_in, w * h * 3);
    cudaMalloc(&d_out, w * h);

    cudaMemcpy(d_in, src.data, w * h * 3, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((w + threads.x - 1) / threads.x, (h + threads.y - 1) / threads.y);
    bgrToGrayKernel<<<blocks, threads>>>(d_in, d_out, w, h);
    cudaDeviceSynchronize();

    cv::Mat cudaGray(h, w, CV_8UC1);
    cudaMemcpy(cudaGray.data, d_out, w * h, cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_out);

    cv::Mat cvGray, diff;
    cv::cvtColor(src, cvGray, cv::COLOR_BGR2GRAY);
    cv::absdiff(cudaGray, cvGray, diff);
    
    std::cout << "Gri Tonlama Donusum Hata Degeri (AbsDiff Sum): " << cv::sum(diff)[0] << std::endl;
    return 0;
}
