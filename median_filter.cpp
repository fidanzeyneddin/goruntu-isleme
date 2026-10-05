#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm> // std::sort için gerekli

// --- MANUEL 5x5 MEDYAN FİLTRESİ ---
void manualMedianFilter5x5(const cv::Mat& src, cv::Mat& dst) {
    // Çıktı matrisini oluştur
    dst = src.clone();
    int rows = src.rows;
    int cols = src.cols;

    // 5x5 maske olduğu için kenarlardan 2'şer piksel boşluk bırakıyoruz (taşmayı önlemek için)
    for (int y = 2; y < rows - 2; ++y) {
        for (int x = 2; x < cols - 2; ++x) {
            
            // 25 elemanlı komşuluk piksellerini tutacak vektör
            std::vector<uchar> window;
            window.reserve(25);

            // 5x5'lik pencereyi merkez etrafında gezdir
            for (int ky = -2; ky <= 2; ++ky) {
                for (int kx = -2; kx <= 2; ++kx) {
                    window.push_back(src.at<uchar>(y + ky, x + kx));
                }
            }

            // Pikselleri küçükten büyüğe sırala
            std::sort(window.begin(), window.end());

            // 25 elemanın tam ortasındaki (medyan) değeri al (indeks 12)
            dst.at<uchar>(y, x) = window[12];
        }
    }
}

int main() {
    // Gürültülü resmi oku (Önceki adımdaki noisy.jpg dosyasını kullanıyoruz)
    cv::Mat noisy_img = cv::imread("saltpepper.jpg", cv::IMREAD_GRAYSCALE);
    
    if (noisy_img.empty()) {
        std::cout << "Hata: 'saltpepper.jpg' bulunamadi! Lutfen dosyanin var oldugundan emin olun." << std::endl;
        return -1;
    }

    // 1. OpenCV HAZIR FONKSİYONU İLE 5x5 MEDYAN FİLTRESİ
    cv::Mat cv_median;
    // cv::medianBlur fonksiyonunda son parametre maske boyutudur (5 = 5x5)
    cv::medianBlur(noisy_img, cv_median, 5);
    cv::imwrite("median_opencv.jpg", cv_median);

    // 2. MANUEL 5x5 MEDYAN FİLTRESİ (KENDİ KODUMUZ)
    cv::Mat man_median;
    manualMedianFilter5x5(noisy_img, man_median);
    cv::imwrite("median_manual.jpg", man_median);

    std::cout << "Gorev tamamlandi! 'median_opencv.jpg' ve 'median_manual.jpg' olusturuldu." << std::endl;

    return 0;
}
