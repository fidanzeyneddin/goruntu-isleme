#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat img = cv::imread("resim.jpg", cv::IMREAD_GRAYSCALE);
    
    if(img.empty()) {
        std::cout << "Hata: resim.jpg bulunamadi!" << std::endl;
        return -1;
    }

    cv::Mat blurred7x7, logResult;

    cv::GaussianBlur(img, blurred7x7, cv::Size(7, 7), 0);

    cv::Laplacian(blurred7x7, logResult, CV_32F, 3);

    cv::convertScaleAbs(logResult, logResult);

    cv::imwrite("log_7x7_sonuc.jpg", logResult);
    
    std::cout << "Islem basarili! Sonuclar 'log_7x7_sonuc.jpg' olarak kaydedildi." << std::endl;

    return 0;
}
