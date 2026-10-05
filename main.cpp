#include <opencv2/opencv.hpp>
#include <iostream>

void contrastStretching(const cv::Mat& inputImage, cv::Mat& outputImage) {
    // 1. Resmi gri seviyeye dönüştür (eğer renkli ise)
    cv::Mat gray;
    if (inputImage.channels() == 3) {
        cv::cvtColor(inputImage, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = inputImage.clone();
    }

    // 2. Minimum ve maksimum piksel değerlerini bul
    double minVal, maxVal;
    cv::minMaxLoc(gray, &minVal, &maxVal);

    std::cout << "Bulunan Min Deger: " << minVal << std::endl;
    std::cout << "Bulunan Max Deger: " << maxVal << std::endl;

    outputImage = cv::Mat::zeros(gray.size(), CV_8UC1);

    if (maxVal == minVal) {
        gray.copyTo(outputImage);
        return;
    }

    // 3. Doğrusal kontrast germe
    for (int i = 0; i < gray.rows; i++) {
        for (int j = 0; j < gray.cols; j++) {
            uchar pixel = gray.at<uchar>(i, j);
            double scaled = (static_cast<double>(pixel) - minVal) * 255.0 / (maxVal - minVal);
            outputImage.at<uchar>(i, j) = cv::saturate_cast<uchar>(scaled);
        }
    }
}

int main() {
    // resim.jpg dosyasını oku
    cv::Mat img = cv::imread("resim.jpg", cv::IMREAD_UNCHANGED);
    
    if (img.empty()) {
        std::cout << "Resim yuklenemedi!" << std::endl;
        return -1;
    }

    cv::Mat result;
    contrastStretching(img, result);

    // Colab'de GUI penceresi açılamadığı için sonucu dosya olarak kaydediyoruz
    cv::imwrite("sonuc.jpg", result);
    std::cout << "Kontrast gerilmis resim 'sonuc.jpg' olarak kaydedildi." << std::endl;

    return 0;
}
