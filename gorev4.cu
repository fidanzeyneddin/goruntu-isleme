#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>

typedef void (*CudaConverterFunc)(const cv::Mat&, cv::Mat&);

__global__ void bgrToRgbKernel(unsigned char* d_in, unsigned char* d_out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) {
        int idx = (y * w + x) * 3;
        d_out[idx] = d_in[idx + 2];
        d_out[idx + 1] = d_in[idx + 1];
        d_out[idx + 2] = d_in[idx];
    }
}

// Wrapper fonksiyon 1
void convertBgrToRgbWrapper(const cv::Mat& src, cv::Mat& dst) {
    dst.create(src.rows, src.cols, CV_8UC3);
    unsigned char *d_in, *d_out;
    cudaMalloc(&d_in, src.total() * 3); cudaMalloc(&d_out, src.total() * 3);
    cudaMemcpy(d_in, src.data, src.total() * 3, cudaMemcpyHostToDevice);
    
    dim3 threads(16, 16);
    dim3 blocks((src.cols + threads.x - 1) / threads.x, (src.rows + threads.y - 1) / threads.y);
    bgrToRgbKernel<<<blocks, threads>>>(d_in, d_out, src.cols, src.rows);
    cudaDeviceSynchronize();
    
    cudaMemcpy(dst.data, d_out, src.total() * 3, cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_out);
}

int main() {
    cv::Mat src = cv::imread("resim.jpg", cv::IMREAD_COLOR);
    if (src.empty()) return -1;

    // Fonksiyon Pointer kullanımı
    CudaConverterFunc ptrToRgb = convertBgrToRgbWrapper;

    cv::Mat customRgb, cvRgb, diffRgb;
    ptrToRgb(src, customRgb);
    cv::cvtColor(src, cvRgb, cv::COLOR_BGR2RGB);
    
    cv::absdiff(customRgb, cvRgb, diffRgb);
    double errorRgb = cv::sum(diffRgb)[0] + cv::sum(diffRgb)[1] + cv::sum(diffRgb)[2];

    std::cout << "Fonksiyon Pointer (CUDA) BGR->RGB Toplam Hata Degeri: " << errorRgb << std::endl;
    return 0;
}
