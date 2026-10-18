#include <opencv2/opencv.hpp>
#include <iostream>

int main(){
    cv::Mat img = cv::imread("resim.jpg",cv::IMREAD_GRAYSCALE);
    cv::Mat blurred, prewittX, prewittY, prewittTotal;

    cv::GaussianBlur(img, blurred, cv::Size(3,3),0);

    cv::Mat kernelX =(cv::Mat_<float>(3,3)<< -1, 0, 1,
                                             -1, 0, 1,
                                             -1, 0, 1);
    
    cv::Mat kernelY =(cv::Mat_<float>(3,3)<< -1, -1, -1,
                                              0, 0, 0,
                                              1, 1, 1);
    cv::filter2D(blurred,prewittX,CV_32F,kernelX);
    cv::filter2D(blurred,prewittY,CV_32F,kernelY);

    cv::convertScaleAbs(prewittX,prewittX);
    cv::convertScaleAbs(prewittY,prewittY);

    cv::addWeighted(prewittX,0.5,prewittY,0.5,0,prewittTotal);

    cv::imwrite("prewitt_sonuc.jpg", prewittTotal);
    
    std::cout << "Islem basarili! Sonuclar 'prewitt_sonuc.jpg' olarak kaydedildi." << std::endl;

    return 0;
    }
