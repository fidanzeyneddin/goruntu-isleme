#include <opencv2/opencv.hpp>
#include <iostream>

void applyMeanFilter(const cv::Mat& src, cv::Mat& dst) {
    // Çıktı resmini başlangıçta orijinal resmin kopyası olarak oluştur
    // Böylece filtre uğramayan en dış kenarlar siyah kalmaz, orijinal kalır.
    dst = src.clone();

    int rows = src.rows;
    int cols = src.cols;

    // Resmin üzerinde piksel piksel geziniyoruz.
    // DİKKAT: 3x3 maske dışarı taşmasın diye x ve y döngülerini 1'den başlatıp sondan 1 eksik bitiriyoruz.
    for (int y = 1; y < rows - 1; ++y) {
        for (int x = 1; x < cols - 1; ++x) {
            
            int sum = 0;

            // 3x3'lük çekirdeği (kernel/maske) merkez pikselin etrafında gezdiriyoruz
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    sum += src.at<uchar>(y + ky, x + kx);
                }
            }

            // Toplamı 9'a bölerek ortalamayı bul ve çıktı resmindeki merkez piksele yaz
            dst.at<uchar>(y, x) = cv::saturate_cast<uchar>(sum / 9);
        }
    }
}

int main() {
    // Önceki adımda oluşturduğumuz gürültülü resmi oku
    cv::Mat noisy_img = cv::imread("noisy.jpg", cv::IMREAD_GRAYSCALE);
    
    if (noisy_img.empty()) {
        std::cout << "Hata: 'noisy.jpg' bulunamadi!" << std::endl;
        return -1;
    }

    cv::Mat filtered_img;
    
    // Filtreyi uygula
    applyMeanFilter(noisy_img, filtered_img);

    // Sonucu kaydet
    cv::imwrite("filtrelenmis_resim.jpg", filtered_img);
    std::cout << "3x3 Ortalama Filtre (Konvolusyon) basariyla uygulandi!" << std::endl;
    std::cout << "Sonuc 'filtrelenmis_resim.jpg' olarak kaydedildi." << std::endl;

    return 0;
}
