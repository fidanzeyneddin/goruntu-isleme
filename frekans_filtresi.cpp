#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <string>

// 1. Dalga ve Gürültü Üreticiler
void generateSineWaveFreq(std::vector<double>& vec, double amplitude, double frequency, double phase) {
    for (size_t t = 0; t < vec.size(); ++t) {
        vec[t] = amplitude * std::sin(2.0 * CV_PI * frequency * t / vec.size() + phase);
    }
}

void addNoise(const std::vector<double>& src, std::vector<double>& dst, double noise_ratio) {
    dst = src;
    double max_amp = 0.0;
    for (double val : src) max_amp = std::max(max_amp, std::abs(val));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-max_amp * noise_ratio, max_amp * noise_ratio);
    for (size_t i = 0; i < dst.size(); ++i) dst[i] += dis(gen);
}

// 2. Frekans Filtresi (İleri DFT -> Eşikleme -> Geri IDFT)
void applyFrequencyFilter(const std::vector<double>& src, std::vector<double>& dst, double threshold_ratio) {
    // a. İleri Dönüşüm (DFT)
    cv::Mat srcMat = cv::Mat(1, src.size(), CV_64F, (void*)src.data()).clone();
    cv::Mat planes[] = {srcMat, cv::Mat::zeros(srcMat.size(), CV_64F)};
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);
    cv::dft(complexI, complexI);
    
    // b. Genlikleri Hesapla ve Maksimumu Bul
    cv::split(complexI, planes);
    cv::Mat mag;
    cv::magnitude(planes[0], planes[1], mag);
    
    double minVal, maxVal;
    cv::minMaxLoc(mag, &minVal, &maxVal);
    double threshold = maxVal * threshold_ratio; // Eşik değerini belirle
    
    // c. Filtreleme: Eşiğin altındaki harmoniklerin hem Reel hem Sanal kısımlarını sıfırla
    for (int i = 0; i < mag.cols; ++i) {
        if (mag.at<double>(0, i) < threshold) {
            planes[0].at<double>(0, i) = 0.0;
            planes[1].at<double>(0, i) = 0.0;
        }
    }
    
    // d. Ters Dönüşüm (IDFT) ile Sinyali Geri Üret
    cv::merge(planes, 2, complexI);
    cv::Mat restored;
    // DFT_SCALE ile genlikleri orijinal boyutuna küçültüyoruz
    cv::idft(complexI, restored, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    
    // Matrisi std::vector'e aktar
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = restored.at<double>(0, i);
    }
}

// 3. Zaman Uzayı Çizim Fonksiyonu (Dinamik Ölçekli)
cv::Mat drawWaveMat(const std::vector<double>& wave, cv::Scalar color, const std::string& title) {
    int height = 300, width = 512;
    cv::Mat plot = cv::Mat::zeros(height, width, CV_8UC3);
    int centerY = height / 2;
    
    double max_val = 0.0;
    for (double v : wave) max_val = std::max(max_val, std::abs(v));
    double scale = (max_val > 0) ? ((height / 2.0 - 20.0) / max_val) : 1.0;
    
    cv::line(plot, cv::Point(0, centerY), cv::Point(width, centerY), cv::Scalar(100, 100, 100), 1);
    for (size_t i = 0; i < wave.size() - 1; ++i) {
        int y1 = centerY - static_cast<int>(wave[i] * scale); 
        int y2 = centerY - static_cast<int>(wave[i+1] * scale);
        y1 = std::max(0, std::min(height - 1, y1));
        y2 = std::max(0, std::min(height - 1, y2));
        cv::line(plot, cv::Point(i, y1), cv::Point(i+1, y2), color, 2);
    }
    cv::putText(plot, title, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
    return plot;
}

int main() {
    int size = 512;
    // Maksimum pikin %20'si altındakileri sil
    // (Toplam sinyalde A=12 pik, %20'si 2.4 yapar. Böylece A=5 olan piki silmeyip koruruz)
    double threshold_ratio = 0.20; 
    
    // --- SİNYAL 1: f=50, A=12 ---
    std::vector<double> s1_clean(size), s1_noisy, s1_restored;
    generateSineWaveFreq(s1_clean, 12.0, 50.0, 0.0);
    addNoise(s1_clean, s1_noisy, 0.3);
    applyFrequencyFilter(s1_noisy, s1_restored, threshold_ratio);
    
    // --- SİNYAL 2: f=30, A=5 ---
    std::vector<double> s2_clean(size), s2_noisy, s2_restored;
    generateSineWaveFreq(s2_clean, 5.0, 30.0, 0.0);
    addNoise(s2_clean, s2_noisy, 0.3);
    applyFrequencyFilter(s2_noisy, s2_restored, threshold_ratio);
    
    // --- TOPLAM SİNYAL: S1 + S2 ---
    std::vector<double> s_sum_noisy(size), s_sum_restored(size);
    for(int i = 0; i < size; ++i) {
        s_sum_noisy[i] = s1_noisy[i] + s2_noisy[i];
    }
    applyFrequencyFilter(s_sum_noisy, s_sum_restored, threshold_ratio);
    
    // --- GÖRSELLEŞTİRME ---
    cv::Mat m1_noisy = drawWaveMat(s1_noisy, cv::Scalar(0, 255, 255), "S1 Gurultulu (f=50, A=12)");
    cv::Mat m1_rest = drawWaveMat(s1_restored, cv::Scalar(0, 255, 0), "S1 Frekans Filtreli (Temiz)");
    cv::Mat c1; cv::hconcat(m1_noisy, m1_rest, c1);
    cv::imwrite("filt_1.png", c1);

    cv::Mat m2_noisy = drawWaveMat(s2_noisy, cv::Scalar(0, 255, 255), "S2 Gurultulu (f=30, A=5)");
    cv::Mat m2_rest = drawWaveMat(s2_restored, cv::Scalar(255, 150, 0), "S2 Frekans Filtreli (Temiz)");
    cv::Mat c2; cv::hconcat(m2_noisy, m2_rest, c2);
    cv::imwrite("filt_2.png", c2);

    cv::Mat m_sum_noisy = drawWaveMat(s_sum_noisy, cv::Scalar(0, 255, 255), "Toplam Sinyal (Gurultulu)");
    cv::Mat m_sum_rest = drawWaveMat(s_sum_restored, cv::Scalar(200, 100, 255), "Toplam Sinyal (Frekans Filtreli)");
    cv::Mat c3; cv::hconcat(m_sum_noisy, m_sum_rest, c3);
    cv::imwrite("filt_3.png", c3);
    
    std::cout << "Frekans filtreleme basariyla uygulandi." << std::endl;
    return 0;
}
