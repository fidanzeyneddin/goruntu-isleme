#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <opencv2/opencv.hpp>

// Önceki görevden sinüs dalgası fonksiyonumuz
void generateSineWave(std::vector<double>& vec, double amplitude, double period, double phase) {
    for (size_t t = 0; t < vec.size(); ++t) {
        vec[t] = amplitude * std::sin((2.0 * CV_PI * t / period) + phase);
    }
}

// 3. Görev: Gürültü ekleme fonksiyonu
void addNoise(std::vector<double>& vec, double noise_ratio) {
    // 1. Vektördeki maksimum genliği (mutlak değer olarak) bulalım
    double max_amp = 0.0;
    for (double val : vec) {
        if (std::abs(val) > max_amp) {
            max_amp = std::abs(val);
        }
    }

    // 2. Rastgele sayı üretici ayarları (C++ <random> kütüphanesi)
    std::random_device rd;
    std::mt19937 gen(rd());

    // Gürültü aralığı: [-maksimum_genlik * oran, +maksimum_genlik * oran]
    std::uniform_real_distribution<double> dis(-max_amp * noise_ratio, max_amp * noise_ratio);

    // 3. Vektördeki her bir elemana üretilen rastgele gürültüyü ekle
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] += dis(gen);
    }
}

int main() {
    int size = 512;
    std::vector<double> wave(size);

    // Temel sinüs dalgasını oluştur (Genlik: 100, Periyot: 128)
    generateSineWave(wave, 100.0, 128.0, 0.0);

    // Fonksiyonu çağırarak %20 oranında (0.2) gürültü ekle
    double noise_ratio = 0.2;
    addNoise(wave, noise_ratio);

    // Görselleştirme (Siyah arkaplan üzerine sarı renkli çizim)
    int height = 300;
    int width = size;
    cv::Mat plot = cv::Mat::zeros(height, width, CV_8UC3);
    int centerY = height / 2;

    // X ekseni
    cv::line(plot, cv::Point(0, centerY), cv::Point(width, centerY), cv::Scalar(100, 100, 100), 1);

    for (int i = 0; i < size - 1; ++i) {
        int y1 = centerY - static_cast<int>(wave[i]);
        int y2 = centerY - static_cast<int>(wave[i+1]);

        // Gürültülü dalgayı belirgin olması için sarı renkle (0, 255, 255) çizelim
        cv::line(plot, cv::Point(i, y1), cv::Point(i+1, y2), cv::Scalar(0, 255, 255), 2);
    }

    cv::imwrite("gurultulu_sinus.png", plot);
    std::cout << "Gurultulu gorsel basariyla kaydedildi." << std::endl;

    return 0;
}
