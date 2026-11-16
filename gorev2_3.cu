#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>

using namespace cv;
using namespace std;

// Görev 3: Pikselleri 2'ye bölen CUDA Çekirdeği
__global__ void dividePixelKernel(uchar* data, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        data[idx] = data[idx] / 2;
    }
}

int main() {
    // ==========================================
    // GÖREV 2: 1000x1000 Resim Açma ve Gri Yapma
    // ==========================================
    // Temsili bir 1000x1000 renkli resim oluşturuyoruz
    Mat img1000(1000, 1000, CV_8UC3, Scalar(0, 128, 255)); 
    imwrite("orijinal_1000.bmp", img1000);
    
    Mat gray1000;
    cvtColor(img1000, gray1000, COLOR_BGR2GRAY);
    imwrite("gri_1000.bmp", gray1000);
    
    cout << "--- GÖREV 2 ANALİZİ ---" << endl;
    cout << "1000x1000 3 Kanallı Boyut (Teorik): " << (1000 * 1000 * 3) + 54 << " Byte" << endl;
    cout << "1000x1000 1 Kanallı (Gri) Boyut (Teorik): " << (1000 * 1000 * 1) + 1024 + 54 << " Byte" << endl;
    // (Gri BMP'lerde genellikle 1024 bytelık bir renk paleti eklenir)

    // ==========================================
    // GÖREV 3: 8K Görüntü (7680x4320) İşleme ve Zaman Ölçümü
    // ==========================================
    cout << "\n--- GÖREV 3 ANALİZİ ---" << endl;
    Mat img8K(4320, 7680, CV_8UC3, Scalar(200, 100, 50));
    int total_bytes = img8K.rows * img8K.cols * img8K.channels();
    
    uchar* d_data;
    cudaMalloc(&d_data, total_bytes);
    cudaMemcpy(d_data, img8K.data, total_bytes, cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = (total_bytes + blockSize - 1) / blockSize;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Yalnızca çekirdeğin (piksel işleminin) çalışma süresini ölçüyoruz
    cudaEventRecord(start);
    dividePixelKernel<<<gridSize, blockSize>>>(d_data, total_bytes);
    cudaEventRecord(stop);
    
    cudaMemcpy(img8K.data, d_data, total_bytes, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    imwrite("islenmis_8K.bmp", img8K);
    
    cout << "8K (3 Kanallı) Toplam Piksel Değeri Sayısı: " << total_bytes << endl;
    cout << "CUDA Çekirdeği Çalışma Süresi: " << milliseconds << " ms" << endl;
    
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
