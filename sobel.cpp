#include <opencv2/opencv.hpp>
#include <iostream>

int main(){
    cv::Mat img=cv::imread("resim.jpg", cv::IMREAD_GRAYSCALE);

    if(img.empty()){
        std::cout<<"Hata:resim.jpg bulunamadi!"<<std::endl;
        return -1;
    }

    cv::Mat blurred, sobelX, sobelY, sobelTotal;

    cv::GaussianBlur(img, blurred, cv::Size(3,3),0);

    cv::Sobel(blurred,sobelX,CV_32F,1,0,3);

    cv::Sobel(blurred,sobelY,CV_32F,0,1,3);

    cv::convertScaleAbs(sobelX,sobelX);
    cv::convertScaleAbs(sobelY,sobelY);

    cv::addWeighted(sobelX,0.5,sobelY,0.5,0,sobelTotal);

    cv::imwrite("sobel_sonuc.jpg",sobelTotal);

    std::cout << "Islem basarili! Sonuclar 'sobel_sonuc.jpg' olarak kaydedildi." << std::endl;

    return 0;
}
