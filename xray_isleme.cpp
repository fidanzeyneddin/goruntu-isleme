#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

int main() {
    std::vector<std::string> resim_isimleri = {
        "resim1.jpg", "resim2.jpg", "resim3.jpg", "resim4.jpg", "resim5.jpg"
    };

    for (int i = 0; i < resim_isimleri.size(); ++i) {
        std::string dosya_adi = resim_isimleri[i];
        
        // Görüntüyü oku
        cv::Mat orijinal = cv::imread(dosya_adi, cv::IMREAD_COLOR);
        if (orijinal.empty()) {
            std::cerr << "Hata: " << dosya_adi << " bulunamadi!" << std::endl;
            continue;
        }

        // 1. Gri tonlamaya dönüştür
        cv::Mat gri_resim;
        cv::cvtColor(orijinal, gri_resim, cv::COLOR_BGR2GRAY);

        // 2. Güvenlik operatörleri için renklendirme (Jet Paleti)
        cv::Mat renklendirilmis;
        cv::applyColorMap(gri_resim, renklendirilmis, cv::COLORMAP_JET);

        // Sonuçları kaydet (Python ile görselleştirmek için)
        std::string gri_kayit = "gri_" + std::to_string(i+1) + ".jpg";
        std::string renkli_kayit = "renkli_" + std::to_string(i+1) + ".jpg";
        
        cv::imwrite(gri_kayit, gri_resim);
        cv::imwrite(renkli_kayit, renklendirilmis);
        
        std::cout << dosya_adi << " islendi ve kaydedildi." << std::endl;
    }

    return 0;
}
