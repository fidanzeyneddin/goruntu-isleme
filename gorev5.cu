#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>

// Görev 3'teki bgrToGrayKernel burada kullanılmak üzere varsayılmıştır.
__global__ void bgrToGrayKernelXray(unsigned char* d_in, unsigned char* d_out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) {
        int c = (y * w + x) * 3; int g = y * w + x;
        d_out[g] = (d_in[c+2] * 4899 + d_in[c+1] * 9617 + d_in[c] * 1868 + 16384) >> 15;
    }
}

int main() {
    std::vector<std::string> resimler = {"resim1.jpg", "resim2.jpg", "resim3.jpg", "resim4.jpg", "resim5.jpg"};
    
    for (const auto& dosya : resimler) {
        cv::Mat xray = cv::imread(dosya, cv::IMREAD_COLOR);
        if (xray.empty()) continue;

        int w = xray.cols, h = xray.rows;
        unsigned char *d_in, *d_out;
        cudaMalloc(&d_in, w * h * 3); cudaMalloc(&d_out, w * h);
        cudaMemcpy(d_in, xray.data, w * h * 3, cudaMemcpyHostToDevice);

        dim3 threads(16, 16);
        dim3 blocks((w + threads.x - 1) / threads.x, (h + threads.y - 1) / threads.y);
        bgrToGrayKernelXray<<<blocks, threads>>>(d_in, d_out, w, h);
        cudaDeviceSynchronize();

        cv::Mat cudaGray(h, w, CV_8UC1);
        cudaMemcpy(cudaGray.data, d_out, w * h, cudaMemcpyDeviceToHost);
        cudaFree(d_in); cudaFree(d_out);

        cv::Mat jetColor;
        cv::applyColorMap(cudaGray, jetColor, cv::COLORMAP_JET);
        
        cv::imwrite("islenmis_" + dosya, jetColor);
        std::cout << dosya << " islendi ve kaydedildi." << std::endl;
    }
    return 0;
}
