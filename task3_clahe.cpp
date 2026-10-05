#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

// --- MANUEL CLAHE GERÇEKLEMESİ ---
void manualCLAHE(const cv::Mat& src, cv::Mat& dst, int gridX = 8, int gridY = 8, float clipLimit = 2.0f) {
    dst = src.clone();
    int width = src.cols;
    int height = src.rows;
    
    // Her bir ızgaranın (tile) genişlik ve yüksekliği
    int tileW = width / gridX;
    int tileH = height / gridY;

    // Her ızgara için Look-Up Table (Eşleme Tablosu) tutacağımız 2 boyutlu vektör
    std::vector<std::vector<int>> luts(gridX * gridY, std::vector<int>(256, 0));

    // 1. ADIM: Her bir ızgara için Histogram, Kırpma (Clipping) ve CDF hesaplama
    for (int ty = 0; ty < gridY; ++ty) {
        for (int tx = 0; tx < gridX; ++tx) {
            int startX = tx * tileW;
            int startY = ty * tileH;
            // Görüntü sonundaki artıkları hesaba katmak için sınır kontrolü
            int endX = (tx == gridX - 1) ? width : startX + tileW;
            int endY = (ty == gridY - 1) ? height : startY + tileH;
            
            int currentTileW = endX - startX;
            int currentTileH = endY - startY;
            int numPixels = currentTileW * currentTileH;

            // Histogram çıkar
            std::vector<int> hist(256, 0);
            for (int y = startY; y < endY; ++y) {
                for (int x = startX; x < endX; ++x) {
                    hist[src.at<uchar>(y, x)]++;
                }
            }

            // Histogramı Sınırla (Clip Limit)
            int clipVal = std::max(1, static_cast<int>(clipLimit * numPixels / 256.0));
            int excess = 0;
            for (int i = 0; i < 256; ++i) {
                if (hist[i] > clipVal) {
                    excess += hist[i] - clipVal;
                    hist[i] = clipVal;
                }
            }

            // Kırpılan fazlalığı diğer piksellere eşit dağıt
            int binIncr = excess / 256;
            int upper = excess % 256;
            for (int i = 0; i < 256; ++i) {
                hist[i] += binIncr;
            }
            if (upper != 0) {
                int step = 256 / upper;
                for (int i = 0; i < upper; ++i) {
                    hist[i * step]++;
                }
            }

            // CDF (Kümülatif Dağılım) ve LUT Hesaplama
            std::vector<int> cdf(256, 0);
            cdf[0] = hist[0];
            for (int i = 1; i < 256; ++i) {
                cdf[i] = cdf[i - 1] + hist[i];
            }

            int tileIdx = ty * gridX + tx;
            for (int i = 0; i < 256; ++i) {
                luts[tileIdx][i] = cv::saturate_cast<uchar>(std::round(static_cast<float>(cdf[i]) / numPixels * 255.0f));
            }
        }
    }

    // 2. ADIM: Bilineer İnterpolasyon ile Pikselleri Güncelleme
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Pikselin merkezine göre ızgara koordinatlarını bul
            float txf = static_cast<float>(x) / tileW - 0.5f;
            float tyf = static_cast<float>(y) / tileH - 0.5f;

            int tx1 = std::floor(txf);
            int ty1 = std::floor(tyf);
            int tx2 = tx1 + 1;
            int ty2 = ty1 + 1;

            float dx = txf - tx1;
            float dy = tyf - ty1;

            // Sınırları aşmamak için kısıtla
            tx1 = std::max(0, std::min(tx1, gridX - 1));
            tx2 = std::max(0, std::min(tx2, gridX - 1));
            ty1 = std::max(0, std::min(ty1, gridY - 1));
            ty2 = std::max(0, std::min(ty2, gridY - 1));

            // Komşu 4 ızgaranın indeksleri
            int idx11 = ty1 * gridX + tx1;
            int idx12 = ty1 * gridX + tx2;
            int idx21 = ty2 * gridX + tx1;
            int idx22 = ty2 * gridX + tx2;

            uchar pixel = src.at<uchar>(y, x);

            // 4 köşeden gelen yeni değerleri al
            float p11 = luts[idx11][pixel];
            float p12 = luts[idx12][pixel];
            float p21 = luts[idx21][pixel];
            float p22 = luts[idx22][pixel];

            // Bilineer interpolasyon formülü
            float val = p11 * (1 - dx) * (1 - dy) +
                        p12 * dx * (1 - dy) +
                        p21 * (1 - dx) * dy +
                        p22 * dx * dy;

            dst.at<uchar>(y, x) = cv::saturate_cast<uchar>(val);
        }
    }
}

int main() {
    // Resmi oku
    cv::Mat img = cv::imread("resim.jpg", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cout << "Resim yuklenemedi!" << std::endl;
        return -1;
    }

    // 1. OpenCV HAZIR FONKSİYONU İLE CLAHE
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(2.0);
    clahe->setTilesGridSize(cv::Size(8, 8));
    
    cv::Mat opencv_clahe_result;
    clahe->apply(img, opencv_clahe_result);
    cv::imwrite("clahe_opencv.jpg", opencv_clahe_result);

    // 2. MANUEL CLAHE FONKSİYONU İLE
    cv::Mat manual_clahe_result;
    manualCLAHE(img, manual_clahe_result, 8, 8, 2.0f);
    cv::imwrite("clahe_manual.jpg", manual_clahe_result);

    std::cout << "Gorev tamamlandi: 'clahe_opencv.jpg' ve 'clahe_manual.jpg' kaydedildi." << std::endl;

    return 0;
}
