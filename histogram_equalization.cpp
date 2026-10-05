#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    // 1. Resmi tek kanallı (grayscale) olarak oku
    cv::Mat img = cv::imread("resim.jpg", cv::IMREAD_GRAYSCALE);

    if (img.empty()) {
        std::cout << "Hata: Resim bulunamadi veya yuklenemedi!" << std::endl;
        return -1;
    }

    int rows = img.rows;
    int cols = img.cols;
    int totalPixels = rows * cols;

    // 2. 256 elemanlı Histogram dizisi oluştur ve hesapla (Hazır fonksiyon yok)
    std::vector<int> hist(256, 0);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            uchar pixelVal = img.at<uchar>(i, j);
            hist[pixelVal]++;
        }
    }

    // 3. Kümülatif Dağılım Fonksiyonu (CDF) Hesaplama (Hazır fonksiyon yok)
    std::vector<int> cdf(256, 0);
    cdf[0] = hist[0];
    for (int i = 1; i < 256; ++i) {
        cdf[i] = cdf[i - 1] + hist[i];
    }

    // Sıfırdan büyük ilk CDF değerini (cdf_min) bul
    int cdf_min = 0;
    for (int i = 0; i < 256; ++i) {
        if (cdf[i] > 0) {
            cdf_min = cdf[i];
            break;
        }
    }

    // 4. Histogram Eşitleme Eşleme Tablosu (Lookup Table / LUT) Oluşturma
    std::vector<uchar> equalized_map(256, 0);
    for (int i = 0; i < 256; ++i) {
        if (totalPixels - cdf_min == 0) {
            equalized_map[i] = i;
        } else {
            // Formül: round((cdf[i] - cdf_min) / (toplam_piksel - cdf_min) * 255)
            float val = static_cast<float>(cdf[i] - cdf_min) / (totalPixels - cdf_min) * 255.0f;
            equalized_map[i] = static_cast<uchar>(std::round(val));
        }
    }

    // 5. Yeni pikselleri eşleme tablosunu kullanarak çıktı resmine ata
    cv::Mat outputImg = cv::Mat::zeros(img.size(), CV_8UC1);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            uchar originalVal = img.at<uchar>(i, j);
            outputImg.at<uchar>(i, j) = equalized_map[originalVal];
        }
    }

    // 6. Çıktı görüntüsünü kaydet
    cv::imwrite("histogram_esitlenmis.jpg", outputImg);
    std::cout << "Histogram esitleme basariyla tamamlandi ve 'histogram_esitlenmis.jpg' olarak kaydedildi." << std::endl;

    return 0;
}
