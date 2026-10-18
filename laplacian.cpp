#include <opencv2/opencv.hpp>
#include <iostream>

int main(){
  cv::Mat img= cv::imread("resim.jpg",cv::IMREAD_GRAYSCALE);

  if(img.empty()){
    std::cout << "Hata:resim.jpg bulunmadi!"<<std::endl;
    return -1;
  }
  cv::Mat blurred, laplacianResult;

  cv::GaussianBlur(img,blurred,cv::Size(3,3),0);

  cv::Laplacian(blurred,laplacianResult,CV_32F,3);

  cv::imwrite("laplacian_sonuc.jpg",laplacianResult);

  std::cout << "Islem basarili! Sonuclar 'laplacian_sonuc.jpg' olarak kaydedildi." << std::endl;

    return 0;
}
