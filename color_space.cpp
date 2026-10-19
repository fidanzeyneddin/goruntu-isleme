#include <opencv2/opencv.hpp>
#include <iostream>

// 1. BGR'den RGB'ye Dönüşüm Fonksiyonu
void convertBgrToRgb(const cv::Mat& src, cv::Mat& dst) {
    dst = cv::Mat(src.rows, src.cols, src.type());
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
            // Kanalları yer değiştir (B, G, R -> R, G, B)
            dst.at<cv::Vec3b>(i, j) = cv::Vec3b(pixel[2], pixel[1], pixel[0]);
        }
    }
}

// 2. BGR'den Grayscale'e Dönüşüm Fonksiyonu
void convertBgrToGray(const cv::Mat& src, cv::Mat& dst) {
    dst = cv::Mat(src.rows, src.cols, CV_8UC1);
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
            
            // OpenCV'nin sıfır hata toleranslı 8-bit tam sayı dönüşüm katsayıları
            // Y = (R*4899 + G*9617 + B*1868 + 16384) >> 15
            int gray = (pixel[2] * 4899 + pixel[1] * 9617 + pixel[0] * 1868 + 16384) >> 15;
            
            dst.at<uchar>(i, j) = cv::saturate_cast<uchar>(gray);
        }
    }
}

// Fonksiyon Pointer'ı (İşaretçisi) Tanımlaması
// Geri dönüş tipi void, parametreleri (const cv::Mat&, cv::Mat&)
typedef void (*ColorConverterFunc)(const cv::Mat&, cv::Mat&);

int main() {
    // Görüntüyü oku
    cv::Mat src = cv::imread("resim.jpg", cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cerr << "Hata: 'resim.jpg' dosyasi bulunamadi! Lutfen Colab dosyalarina yuklediginizden emin olun." << std::endl;
        return -1;
    }

    // Fonksiyon pointer'larına fonksiyonları atama
    ColorConverterFunc ptrToRgb = convertBgrToRgb;
    ColorConverterFunc ptrToGray = convertBgrToGray;

    // Hedef Matrisler
    cv::Mat customRgb, customGray;
    cv::Mat cvRgb, cvGray;

    // Kendi fonksiyonlarımız (Pointer üzerinden çağırma)
    ptrToRgb(src, customRgb);
    ptrToGray(src, customGray);

    // OpenCV kütüphane fonksiyonları
    cv::cvtColor(src, cvRgb, cv::COLOR_BGR2RGB);
    cv::cvtColor(src, cvGray, cv::COLOR_BGR2GRAY);

    // Hata kontrolü için mutlak fark (absolute difference) hesaplama
    cv::Mat diffRgb, diffGray;
    cv::absdiff(customRgb, cvRgb, diffRgb);
    cv::absdiff(customGray, cvGray, diffGray);

    // Farkların toplamı beklenen değer: 0
    double errorRgb = cv::sum(diffRgb)[0] + cv::sum(diffRgb)[1] + cv::sum(diffRgb)[2];
    double errorGray = cv::sum(diffGray)[0];

    std::cout << "BGR -> RGB Donusumu Toplam Hata Degeri: " << errorRgb << std::endl;
    std::cout << "BGR -> GRAY Donusumu Toplam Hata Degeri: " << errorGray << std::endl;

    if (errorRgb == 0 && errorGray == 0) {
        std::cout << "Basarili! Kendi fonksiyonlariniz ile OpenCV ciktilari 100% uyusuyor." << std::endl;
    }
    // Üretilen matrisleri dosyaya kaydetme
    cv::imwrite("sonuc_gray.jpg", customGray);
    
    // Not: OpenCV cv::imwrite varsayılan olarak BGR formatı bekler. 
    // RGB matrisini diske kaydedersek renkler ters (Mavi ve Kırmızı yer değiştirmiş) çıkar.
    // Bu yüzden RGB sonucumuzu kaydederken tekrar BGR'ye çevirip öyle kaydediyoruz.
    cv::Mat reversedForSave;
    cv::cvtColor(customRgb, reversedForSave, cv::COLOR_RGB2BGR);
    cv::imwrite("sonuc_rgb.jpg", reversedForSave);
    return 0;
}
