#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>

using namespace cv;
using namespace std;

// BGR uzayında kural tabanlı ten tespiti Kernel'i
__global__ void rgbSkinDetectKernel(uchar* bgr_in, uchar* bgr_out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        uchar b = bgr_in[idx];
        uchar g = bgr_in[idx+1];
        uchar r = bgr_in[idx+2];
        
        // GPU içinde max ve min bulma işlemleri
        uchar max_val = r > g ? (r > b ? r : b) : (g > b ? g : b);
        uchar min_val = r < g ? (r < b ? r : b) : (g < b ? g : b);
        int r_g_diff = r > g ? (r - g) : (g - r); // abs(r-g) alternatifi
        
        // Klasik RGB ten rengi kuralı
        bool isSkin = (r > 95 && g > 40 && b > 20 &&
                       (max_val - min_val > 15) &&
                       r_g_diff > 15 && r > g && r > b);
        
        if (isSkin) {
            bgr_out[idx] = b; bgr_out[idx+1] = g; bgr_out[idx+2] = r;
        } else { // Ten değilse Tam Kırmızı (BGR: 0, 0, 255)
            bgr_out[idx] = 0; bgr_out[idx+1] = 0; bgr_out[idx+2] = 255;
        }
    }
}

// HSV ve YCbCr için ten tespiti Kernel'i
__global__ void hsvYcbcrSkinDetectKernel(uchar* orig_bgr, uchar* color_space, uchar* out_bgr, int width, int height, int space_type) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        uchar ch1 = color_space[idx];
        uchar ch2 = color_space[idx+1];
        uchar ch3 = color_space[idx+2];
        
        bool isSkin = false;
        if (space_type == 1) { 
            // HSV Kuralları (uchar zaten >= 0 ve <= 255 olduğu için gereksiz kontroller çıkarıldı)
            if (ch1 <= 20 && ch2 >= 48 && ch3 >= 80) isSkin = true;
        } else if (space_type == 2) { 
            // YCbCr Kuralları
            if (ch2 >= 133 && ch2 <= 173 && ch3 >= 77 && ch3 <= 127) isSkin = true;
        }
        
        if (isSkin) {
            out_bgr[idx] = orig_bgr[idx]; out_bgr[idx+1] = orig_bgr[idx+1]; out_bgr[idx+2] = orig_bgr[idx+2];
        } else {
            out_bgr[idx] = 0; out_bgr[idx+1] = 0; out_bgr[idx+2] = 255;
        }
    }
}

int main() {
    Mat bgr = imread("insan_resmi.bmp");
    if(bgr.empty()) { cout << "HATA: insan_resmi.bmp bulunamadi!" << endl; return -1; }
    
    int w = bgr.cols;
    int h = bgr.rows;
    int bytes = w * h * 3;

    // Farklı uzaylara dönüşüm ve kayıt
    Mat hsv, ycrcb, lab;
    cvtColor(bgr, hsv, COLOR_BGR2HSV);
    cvtColor(bgr, ycrcb, COLOR_BGR2YCrCb);
    cvtColor(bgr, lab, COLOR_BGR2Lab);
    
    imwrite("1_hsv_uzayi.bmp", hsv);
    imwrite("2_yuv_uzayi.bmp", ycrcb); 
    imwrite("3_lab_uzayi.bmp", lab);

    // CUDA Bellek Ayırma
    uchar *d_bgr, *d_hsv, *d_ycrcb, *d_out;
    cudaMalloc(&d_bgr, bytes);
    cudaMalloc(&d_hsv, bytes);
    cudaMalloc(&d_ycrcb, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_bgr, bgr.data, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hsv, hsv.data, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ycrcb, ycrcb.data, bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    Mat result(h, w, CV_8UC3);

    // 1. RGB Ten Tespiti
    rgbSkinDetectKernel<<<grid, block>>>(d_bgr, d_out, w, h);
    cudaMemcpy(result.data, d_out, bytes, cudaMemcpyDeviceToHost);
    imwrite("4_rgb_ten_tespiti.bmp", result);

    // 2. HSV Ten Tespiti 
    hsvYcbcrSkinDetectKernel<<<grid, block>>>(d_bgr, d_hsv, d_out, w, h, 1);
    cudaMemcpy(result.data, d_out, bytes, cudaMemcpyDeviceToHost);
    imwrite("5_hsv_ten_tespiti.bmp", result);

    // 3. YCbCr Ten Tespiti
    hsvYcbcrSkinDetectKernel<<<grid, block>>>(d_bgr, d_ycrcb, d_out, w, h, 2);
    cudaMemcpy(result.data, d_out, bytes, cudaMemcpyDeviceToHost);
    imwrite("6_ycrcb_ten_tespiti.bmp", result);

    cudaFree(d_bgr); cudaFree(d_hsv); cudaFree(d_ycrcb); cudaFree(d_out);
    cout << "Tüm dönüşümler ve ten tespiti işlemleri GPU üzerinde tamamlandi!" << endl;
    return 0;
}
