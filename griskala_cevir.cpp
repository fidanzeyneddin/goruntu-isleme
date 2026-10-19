#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

// Kendi yazdığımız RGB'den Gri Tonlamaya çevirme fonksiyonu
void customCvtColor(const cv::Mat& src, cv::Mat& dst) {
    // Hedef görüntüyü aynı boyutta ve tek kanallı (CV_8UC1) olarak oluştur
    dst.create(src.rows, src.cols, CV_8UC1);

    for (int r = 0; r < src.rows; ++r) {
        for (int c = 0; c < src.cols; ++c) {
            // Görüntü BGR formatında olduğu için kanalları sırasıyla okuyoruz
            cv::Vec3b pixel = src.at<cv::Vec3b>(r, c);
            uchar B = pixel[0];
            uchar G = pixel[1];
            uchar R = pixel[2];

            // Standart parlaklık formülü: 0.299*R + 0.587*G + 0.114*B
            // saturate_cast, sonucun 0-255 sınırları içinde kalmasını sağlar
            dst.at<uchar>(r, c) = cv::saturate_cast<uchar>(0.299 * R + 0.587 * G + 0.114 * B);
        }
    }
}

int main() {
    // Görüntüyü oku
    cv::Mat img = cv::imread("resim.jpg");
    if (img.empty()) {
        std::cout << "Goruntu bulunamadi! 'goruntu.jpg' dosyasini kontrol edin." << std::endl;
        return -1;
    }

    cv::Mat gray_cv, gray_custom;

    // 1. OpenCV'nin hazır fonksiyonu ile çevirme
    cv::cvtColor(img, gray_cv, cv::COLOR_BGR2GRAY);

    // 2. Kendi yazdığımız fonksiyon ile çevirme
    customCvtColor(img, gray_custom);

    // 3. İki görüntü arasındaki mutlak farkların toplamını (Hata Değeri) hesapla
    long long total_error = 0;
    for (int r = 0; r < img.rows; ++r) {
        for (int c = 0; c < img.cols; ++c) {
            int cv_val = gray_cv.at<uchar>(r, c);
            int custom_val = gray_custom.at<uchar>(r, c);
            
            // İki piksel arasındaki farkın mutlak değerini al ve toplama ekle
            total_error += std::abs(cv_val - custom_val);
        }
    }

    // Sonuçları ekrana yazdır
    std::cout << "--- HESAPLAMA SONUCLARI ---" << std::endl;
    std::cout << "Goruntu Boyutu: " << img.cols << "x" << img.rows << " piksel" << std::endl;
    std::cout << "Toplam Mutlak Hata Degeri (SAD): " << total_error << std::endl;

    // Görüntüleri Colab'de Python ile gösterebilmek için diske kaydet
    cv::imwrite("gri_cv.jpg", gray_cv);
    cv::imwrite("gri_custom.jpg", gray_custom);
    std::cout << "Goruntuler 'gri_cv.jpg' ve 'gri_custom.jpg' olarak kaydedildi." << std::endl;

    return 0;
}
