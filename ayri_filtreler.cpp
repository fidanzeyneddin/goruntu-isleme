#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <string>

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

    for (size_t i = 0; i < dst.size(); ++i) {
        dst[i] += dis(gen);
    }
}

// 3. 1 Boyutlu Medyan Filtresi
void apply1DMedianFilter(const std::vector<double>& src, std::vector<double>& dst, int ksize) {
    dst.resize(src.size());
    int pad = ksize / 2;
    for (int i = 0; i < (int)src.size(); ++i) {
        std::vector<double> window;
        for (int j = -pad; j <= pad; ++j) {
            int idx = std::max(0, std::min((int)src.size() - 1, i + j));
            window.push_back(src[idx]);
        }
        std::sort(window.begin(), window.end());
        dst[i] = window[window.size() / 2];
    }
}

// 4. Verilen vektörü ayrı bir resim dosyası olarak kaydetme fonksiyonu
void saveWaveAsImage(const std::vector<double>& wave, const std::string& filename, cv::Scalar color) {
    int height = 300;
    int width = wave.size();
    cv::Mat plot = cv::Mat::zeros(height, width, CV_8UC3);
    int centerY = height / 2;

    // X ekseni çizgisi
    cv::line(plot, cv::Point(0, centerY), cv::Point(width, centerY), cv::Scalar(100, 100, 100), 1);

    // Dalgayı çizdirme
    for (size_t i = 0; i < wave.size() - 1; ++i) {
        int y1 = centerY - static_cast<int>(wave[i]);
        int y2 = centerY - static_cast<int>(wave[i+1]);
        cv::line(plot, cv::Point(i, y1), cv::Point(i+1, y2), color, 2);
    }

    cv::imwrite(filename, plot);
    std::cout << filename << " dosyasi basariyla kaydedildi." << std::endl;
}

int main() {
    int size = 512;

    // Temiz veriyi oluştur ve kaydet (Yeşil)
    std::vector<double> clean_wave(size);
    generateSineWave(clean_wave, 100.0, 128.0, 0.0);
    saveWaveAsImage(clean_wave, "1_temiz_sinus.png", cv::Scalar(0, 255, 0));

    // Gürültülü veriyi oluştur ve kaydet (Sarı)
    std::vector<double> noisy_wave;
    addNoise(clean_wave, noisy_wave, 0.3); // %30 gürültü
    saveWaveAsImage(noisy_wave, "2_gurultulu_sinus.png", cv::Scalar(0, 255, 255));

    // --- GAUSS FİLTRESİ UYGULAMASI VE KAYDI ---
    cv::Mat signalMat(1, size, CV_64F, noisy_wave.data());
    cv::Mat gaussMat;
    cv::GaussianBlur(signalMat, gaussMat, cv::Size(15, 1), 0); // Kernel: 15
    std::vector<double> waveGauss(gaussMat.begin<double>(), gaussMat.end<double>());
    saveWaveAsImage(waveGauss, "3_gauss_sinus.png", cv::Scalar(255, 150, 0)); // Turuncumsu mavi (OpenCV BGR kullanır, bu renk BGR: 255,150,0)

    // --- MEDYAN FİLTRESİ UYGULAMASI VE KAYDI ---
    std::vector<double> waveMedian;
    apply1DMedianFilter(noisy_wave, waveMedian, 15); // Kernel: 15
    saveWaveAsImage(waveMedian, "4_medyan_sinus.png", cv::Scalar(0, 0, 255)); // Kırmızı

    return 0;
}
