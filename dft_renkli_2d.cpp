#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>

// 1. 2'nin kuvvetine yuvarlama ve kırpma işlemleri
int getClosestPowerOf2(int n) { 
    return 1 << static_cast<int>(std::floor(std::log2(n))); 
}

cv::Mat cropToPowerOf2(const cv::Mat& img) {
    int new_width = getClosestPowerOf2(img.cols);
    int new_height = getClosestPowerOf2(img.rows);
    return img(cv::Rect(0, 0, new_width, new_height)).clone();
}

// 2. 2-Boyutlu DFT Fonksiyonu
cv::Mat compute2D_DFT(const cv::Mat& src) {
    cv::Mat padded; 
    src.convertTo(padded, CV_32F);
    
    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexI; 
    cv::merge(planes, 2, complexI);
    
    cv::dft(complexI, complexI);
    
    cv::split(complexI, planes);
    cv::Mat magI; 
    cv::magnitude(planes[0], planes[1], magI); // Harmonik (Genlik) matrisini hesapla
    return magI;
}

// 3. Harmonik Matrisini Renklendirme ve Görselleştirme
cv::Mat processSpectrumForVisualizationColored(cv::Mat& magI) {
    // Logaritmik dönüşüm
    magI += cv::Scalar::all(1);
    cv::log(magI, magI);
    
    // Merkez kaydırma (Düşük frekansları ortaya alma)
    magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));
    int cx = magI.cols / 2, cy = magI.rows / 2;
    cv::Mat q0(magI, cv::Rect(0, 0, cx, cy)), q1(magI, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(magI, cv::Rect(0, cy, cx, cy)), q3(magI, cv::Rect(cx, cy, cx, cy)), tmp;
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);
    
    // 0-255 aralığına çekme ve Renk Haritası (JET) uygulama
    cv::normalize(magI, magI, 0, 1, cv::NORM_MINMAX);
    cv::Mat spectrum8U; 
    magI.convertTo(spectrum8U, CV_8U, 255.0);
    
    cv::Mat coloredSpectrum;
    cv::applyColorMap(spectrum8U, coloredSpectrum, cv::COLORMAP_JET);
    
    return coloredSpectrum;
}

int main() {
    cv::Mat original = cv::imread("orman_web.jpg", cv::IMREAD_GRAYSCALE);
    if(original.empty()) {
        std::cout << "Hata: 'orman_web.jpg' bulunamadi." << std::endl;
        return -1;
    }
    
    // İşlemler
    cv::Mat cropped = cropToPowerOf2(original);
    cv::Mat magnitude_spectrum = compute2D_DFT(cropped);
    cv::Mat colored_spectrum = processSpectrumForVisualizationColored(magnitude_spectrum);
    
    // Kayıt
    cv::imwrite("1_kirpilmis_orijinal.jpg", cropped);
    cv::imwrite("2_renkli_spektrum.jpg", colored_spectrum);
    
    std::cout << "Renkli spektrum islemi basariyla tamamlandi." << std::endl;
    return 0;
}
