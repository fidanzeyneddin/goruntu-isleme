#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <string>

// 1. Periyot yerine doğrudan Frekans alacak şekilde güncellenmiş dalga üretici
void generateSineWaveFreq(std::vector<double>& vec, double amplitude, double frequency, double phase) {
    for (size_t t = 0; t < vec.size(); ++t) {
        // Frekans: Vektör boyutu boyunca kaç adet tam dalga sığacağı (Örn: 50)
        vec[t] = amplitude * std::sin(2.0 * CV_PI * frequency * t / vec.size() + phase);
    }
}

// 2. Gürültü Ekleme Fonksiyonu
void addNoise(const std::vector<double>& src, std::vector<double>& dst, double noise_ratio) {
    dst = src;
    double max_amp = 0.0;
    for (double val : src) max_amp = std::max(max_amp, std::abs(val));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-max_amp * noise_ratio, max_amp * noise_ratio);
    for (size_t i = 0; i < dst.size(); ++i) dst[i] += dis(gen);
}

// 3. DFT Fonksiyonu
std::vector<double> computeDFT(const std::vector<double>& src) {
    cv::Mat srcMat = cv::Mat(1, src.size(), CV_64F, (void*)src.data()).clone();
    cv::Mat planes[] = {srcMat, cv::Mat::zeros(srcMat.size(), CV_64F)};
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);
    cv::dft(complexI, complexI);
    cv::split(complexI, planes);
    cv::magnitude(planes[0], planes[1], planes[0]);
    
    int half_N = src.size() / 2;
    std::vector<double> harmonics(half_N);
    for (int i = 0; i < half_N; ++i) {
        harmonics[i] = planes[0].at<double>(0, i) / (src.size() / 2.0);
    }
    return harmonics;
}

// 4. Zaman Uzayı Grafiği (Dinamik Ölçeklendirme Eklendi)
cv::Mat drawWaveMat(const std::vector<double>& wave, cv::Scalar color, const std::string& title) {
    int height = 300, width = 512;
    cv::Mat plot = cv::Mat::zeros(height, width, CV_8UC3);
    int centerY = height / 2;
    
    // Genlikler 5 veya 12 gibi küçük değerler olduğu için grafiği ekrana sığdırmak adına büyütüyoruz
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

// 5. Frekans Uzayı Grafiği (Dinamik Ölçeklendirme Eklendi)
cv::Mat drawSpectrumMat(const std::vector<double>& spectrum, cv::Scalar color, const std::string& title) {
    int height = 300, width = 512;
    cv::Mat plot = cv::Mat::zeros(height, width, CV_8UC3);
    int base_Y = height - 20;
    
    double max_val = 0.0;
    for (double v : spectrum) max_val = std::max(max_val, std::abs(v));
    double scale = (max_val > 0) ? ((height - 40.0) / max_val) : 1.0;
    
    cv::line(plot, cv::Point(0, base_Y), cv::Point(width, base_Y), cv::Scalar(100, 100, 100), 1);
    for (size_t i = 0; i < spectrum.size(); ++i) {
        int x = i * 2; 
        int y = base_Y - static_cast<int>(spectrum[i] * scale); 
        y = std::max(0, std::min(height - 1, y));
        cv::line(plot, cv::Point(x, base_Y), cv::Point(x, y), color, 2);
    }
    cv::putText(plot, title, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
    return plot;
}

int main() {
    int size = 512;
    
    // --- SİNYAL 1: Frekans = 50, Genlik = 12 ---
    std::vector<double> s1_clean(size), s1_noisy;
    generateSineWaveFreq(s1_clean, 12.0, 50.0, 0.0);
    addNoise(s1_clean, s1_noisy, 0.3); // %30 gürültü eklendi
    std::vector<double> dft1 = computeDFT(s1_noisy);
    
    // --- SİNYAL 2: Frekans = 30, Genlik = 5 ---
    std::vector<double> s2_clean(size), s2_noisy;
    generateSineWaveFreq(s2_clean, 5.0, 30.0, 0.0);
    addNoise(s2_clean, s2_noisy, 0.3); // %30 gürültü eklendi
    std::vector<double> dft2 = computeDFT(s2_noisy);
    
    // --- TOPLAM SİNYAL: S1 + S2 ---
    std::vector<double> s_sum(size);
    for(int i = 0; i < size; ++i) {
        s_sum[i] = s1_noisy[i] + s2_noisy[i];
    }
    std::vector<double> dft_sum = computeDFT(s_sum);
    
    // --- GÖRSELLEŞTİRME VE KAYIT ---
    cv::Mat m_s1 = drawWaveMat(s1_noisy, cv::Scalar(0, 255, 0), "S1 Zaman: f=50, A=12");
    cv::Mat m_dft1 = drawSpectrumMat(dft1, cv::Scalar(0, 255, 0), "S1 DFT Spektrumu");
    cv::Mat c1; cv::hconcat(m_s1, m_dft1, c1);
    cv::imwrite("1_sinyal_f50.png", c1);

    cv::Mat m_s2 = drawWaveMat(s2_noisy, cv::Scalar(255, 150, 0), "S2 Zaman: f=30, A=5");
    cv::Mat m_dft2 = drawSpectrumMat(dft2, cv::Scalar(255, 150, 0), "S2 DFT Spektrumu");
    cv::Mat c2; cv::hconcat(m_s2, m_dft2, c2);
    cv::imwrite("2_sinyal_f30.png", c2);

    cv::Mat m_sum = drawWaveMat(s_sum, cv::Scalar(0, 255, 255), "Toplam Sinyal (S1 + S2)");
    cv::Mat m_dftsum = drawSpectrumMat(dft_sum, cv::Scalar(0, 255, 255), "Toplam DFT Spektrumu");
    cv::Mat c_sum; cv::hconcat(m_sum, m_dftsum, c_sum);
    cv::imwrite("3_sinyal_toplam.png", c_sum);
    
    std::cout << "Analiz tamamlandi ve gorseller kaydedildi." << std::endl;
    return 0;
}
