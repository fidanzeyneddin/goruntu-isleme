#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>

// 1. Kırpma ve Gürültü Fonksiyonları
int getClosestPowerOf2(int n) { return 1 << static_cast<int>(std::floor(std::log2(n))); }

cv::Mat cropToPowerOf2(const cv::Mat& img) {
    int new_width = getClosestPowerOf2(img.cols);
    int new_height = getClosestPowerOf2(img.rows);
    return img(cv::Rect(0, 0, new_width, new_height)).clone();
}

void addGaussianNoise(const cv::Mat& src, cv::Mat& dst, double mean = 0.0, double stddev = 30.0) {
    cv::Mat noise(src.size(), CV_32F);
    cv::randn(noise, mean, stddev);
    cv::Mat src_float; src.convertTo(src_float, CV_32F);
    cv::Mat dst_float = src_float + noise;
    dst_float.convertTo(dst, CV_8U);
}

// 2. Filtre Tipleri
enum FilterType { LOW_PASS, HIGH_PASS, BAND_PASS, BAND_REJECT };

// 3. Frekans Uzayında Filtreleme ve IDFT
cv::Mat applyFilterAndIDFT(const cv::Mat& src, FilterType type, double d0, double d1 = 0, double d2 = 0) {
    cv::Mat padded; src.convertTo(padded, CV_32F);
    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexI; cv::merge(planes, 2, complexI);
    cv::dft(complexI, complexI);
    
    complexI = complexI(cv::Rect(0, 0, complexI.cols & -2, complexI.rows & -2));
    int cx = complexI.cols / 2;
    int cy = complexI.rows / 2;
    cv::Mat q0(complexI, cv::Rect(0, 0, cx, cy)), q1(complexI, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(complexI, cv::Rect(0, cy, cx, cy)), q3(complexI, cv::Rect(cx, cy, cx, cy)), tmp;
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);
    
    for (int y = 0; y < complexI.rows; y++) {
        for (int x = 0; x < complexI.cols; x++) {
            double D = std::sqrt(std::pow(x - cx, 2) + std::pow(y - cy, 2));
            bool keep = true;
            
            if (type == LOW_PASS) keep = (D <= d0);
            else if (type == HIGH_PASS) keep = (D > d0);
            else if (type == BAND_PASS) keep = (D >= d1 && D <= d2);
            else if (type == BAND_REJECT) keep = (D < d1 || D > d2);
            
            if (!keep) {
                complexI.at<cv::Vec2f>(y, x)[0] = 0.0f; 
                complexI.at<cv::Vec2f>(y, x)[1] = 0.0f; 
            }
        }
    }
    
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);
    
    cv::Mat inverseTransform, restored;
    cv::idft(complexI, inverseTransform, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    cv::normalize(inverseTransform, restored, 0, 255, cv::NORM_MINMAX, CV_8U);
    return restored;
}

void drawText(cv::Mat& img, const std::string& text) {
    cv::putText(img, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
}

int main() {
    cv::Mat original = cv::imread("resim.jpeg", cv::IMREAD_GRAYSCALE);
    if(original.empty()) return -1;
    
    cv::Mat cropped = cropToPowerOf2(original);
    
    cv::Mat noisy;
    addGaussianNoise(cropped, noisy, 0.0, 30.0);
    
    // Görüntüleri renkli formata SADECE çizim/kayıt işlemi için ayrı değişkenlere kopyalıyoruz
    cv::Mat cropped_color, noisy_color;
    cv::cvtColor(cropped, cropped_color, cv::COLOR_GRAY2BGR);
    cv::cvtColor(noisy, noisy_color, cv::COLOR_GRAY2BGR);
    drawText(cropped_color, "1. Orijinal");
    drawText(noisy_color, "2. Gurultulu");
    
    // Filtreleme işlemlerinde tek kanallı "noisy" kullanılıyor
    cv::Mat lpf = applyFilterAndIDFT(noisy, LOW_PASS, 60.0);
    cv::Mat lpf_color; cv::cvtColor(lpf, lpf_color, cv::COLOR_GRAY2BGR);
    drawText(lpf_color, "3. Low-Pass (D0=60)");
    
    cv::Mat hpf = applyFilterAndIDFT(noisy, HIGH_PASS, 60.0);
    cv::Mat hpf_color; cv::cvtColor(hpf, hpf_color, cv::COLOR_GRAY2BGR);
    drawText(hpf_color, "4. High-Pass (D0=60)");
    
    cv::Mat bpf = applyFilterAndIDFT(noisy, BAND_PASS, 0.0, 60.0, 120.0);
    cv::Mat bpf_color; cv::cvtColor(bpf, bpf_color, cv::COLOR_GRAY2BGR);
    drawText(bpf_color, "5. Band-Pass (60-120)");
    
    cv::Mat brf = applyFilterAndIDFT(noisy, BAND_REJECT, 0.0, 60.0, 120.0);
    cv::Mat brf_color; cv::cvtColor(brf, brf_color, cv::COLOR_GRAY2BGR);
    drawText(brf_color, "6. Band-Reject (60-120)");
    
    cv::imwrite("1_orijinal_gurultulu.jpg", noisy_color);
    cv::imwrite("2_lpf.jpg", lpf_color);
    cv::imwrite("3_hpf.jpg", hpf_color);
    cv::imwrite("4_bpf.jpg", bpf_color);
    cv::imwrite("5_brf.jpg", brf_color);
    
    std::cout << "Tum filtreler uygulandi ve kaydedildi." << std::endl;
    return 0;
}
