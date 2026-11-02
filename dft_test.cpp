#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <opencv2/opencv.hpp>

// 1. Temiz sinüs dalgası
void generateSineWave(std::vector<double>& vec, double amplitude, double period, double phase) {
    for (size_t t = 0; t < vec.size(); ++t) {
        vec[t] = amplitude * std::sin((2.0 * CV_PI * t / period) + phase);
    }
}

// 2. Gürültü ekleme
void addNoise(const std::vector<double>& src, std::vector<double>& dst, double noise_ratio) {
    dst = src;
    double max_amp = 0.0;
    for (double val : src) max_amp = std::max(max_amp, std::abs(val));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-max_amp * noise_ratio, max_amp * noise_ratio);

    for (size_t i = 0; i < dst.size(); ++i) dst[i] += dis(gen);
}

// 3. DFT Fonksiyonu (Harmonik Vektörünü Döndürür)
std::vector<double> computeDFT(const std::vector<double>& src) {
    // 1D vektörü OpenCV matrisine çevir
    cv::Mat srcMat(1, src.size(), CV_64F, (void*)src.data());
    
    // DFT karmaşık sayılarla (reel + sanal) çalışır.
    // Bizim verimiz tamamen reel, bu yüzden sanal kısmı 0 olan iki kanallı bir matris oluşturuyoruz.
    cv::Mat planes[] = {cv::Mat_<double>(srcMat), cv::Mat::zeros(srcMat.size(), CV_64F)};
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);
    
    // OpenCV DFT işlemi
    cv::dft(complexI, complexI);
    
    // Reel ve sanal kısımları ayır ve genliği (magnitude) hesapla
    // Genlik = sqrt(Re^2 + Im^2)
    cv::split(complexI, planes);
    cv::magnitude(planes[0], planes[1], planes[0]);
    
    // DFT simetrik sonuç üretir. Gerçek sinyallerde frekans bilgisinin ilk yarısı yeterlidir (Nyquist Teoremi).
    // Vektör boyutunun yarısı kadar (N/2) elemanı harmonik vektörü olarak alıyoruz.
    int half_N = src.size() / 2;
    std::vector<double> harmonics(half_N);
    
    for (int i = 0; i < half_N; ++i) {
        // Genlikleri vektörün boyutuna bölerek normalize ediyoruz
        harmonics[i] = planes[0].at<double>(0, i) / (src.size() / 2.0);
    }
    
    return harmonics;
}

// 4. Harmonik Vektörünü (Frekans Spektrumunu) Görselleştirme Çizimi
void saveSpectrumAsImage(const std::vector<double>& spectrum, const std::string& filename, cv::Scalar color) {
    int height = 300;
    int width = spectrum.size() * 2; // Çubukları daha rahat görmek için genişliği artırdık
    cv::Mat plot = cv::Mat::zeros(height, width, CV_8UC3);
    
    // Alt eksen (0 noktası)
    int base_Y = height - 10;
    cv::line(plot, cv::Point(0, base_Y), cv::Point(width, base_Y), cv::Scalar(255, 255, 255), 1);
    
    for (size_t i = 0; i < spectrum.size(); ++i) {
        int x = i * 2;
        // Spektrumdaki değeri ekranda sığacak şekilde ölçeklendir
        int y = base_Y - static_cast<int>(spectrum[i]); 
        if (y < 0) y = 0;
        
        // Çubuk grafik (stem plot) şeklinde dikey çizgiler çek
        cv::line(plot, cv::Point(x, base_Y), cv::Point(x, y), color, 2);
    }
    
    cv::imwrite(filename, plot);
    std::cout << filename << " dosyasi basariyla kaydedildi." << std::endl;
}

int main() {
    int size = 512;
    
    // Verileri oluştur
    std::vector<double> clean_wave(size);
    generateSineWave(clean_wave, 100.0, 128.0, 0.0);
    
    std::vector<double> noisy_wave;
    addNoise(clean_wave, noisy_wave, 0.4); // %40 Gürültü
    
    // Her ikisi için DFT uygula
    std::vector<double> clean_harmonics = computeDFT(clean_wave);
    std::vector<double> noisy_harmonics = computeDFT(noisy_wave);
    
    // Görüntüleri kaydet (Temiz için Yeşil, Gürültülü için Sarı)
    saveSpectrumAsImage(clean_harmonics, "dft_temiz.png", cv::Scalar(0, 255, 0));
    saveSpectrumAsImage(noisy_harmonics, "dft_gurultulu.png", cv::Scalar(0, 255, 255));
    
    return 0;
}
