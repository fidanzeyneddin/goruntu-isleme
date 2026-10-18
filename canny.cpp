#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat img = cv::imread("resim.jpg", cv::IMREAD_GRAYSCALE);
    
    if(img.empty()) {
        std::cout << "Hata: resim.jpg bulunamadi!" << std::endl;
        return -1;
    }

    cv::Mat cannyResult;
    
    cv::Canny(img, cannyResult, 50, 150);

    cv::imwrite("canny_sonuc.jpg", cannyResult);
    
    std::cout << "Islem basarili! Sonuclar 'canny_sonuc.jpg' olarak kaydedildi." << std::endl;

    return 0;
}
