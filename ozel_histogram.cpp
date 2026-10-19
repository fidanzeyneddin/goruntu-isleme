#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

// Kendi yazdığımız histogram hesaplama fonksiyonu
void customCalcHist(const cv::Mat& channelImg, std::vector<int>& histogram) {
    // 0-255 arası değerler için 256 elemanlı diziyi 0 ile dolduruyoruz
    histogram.assign(256, 0); 
    
    // Görüntünün matrisindeki satır ve sütunları tek tek geziyoruz
    for (int r = 0; r < channelImg.rows; ++r) {
        for (int c = 0; c < channelImg.cols; ++c) {
            // O anki pikselin 0-255 arasındaki yoğunluk değerini al
            uchar pixelValue = channelImg.at<uchar>(r, c);
            
            // İlgili yoğunluk değerinin sayısını 1 artır
            histogram[pixelValue]++;
        }
    }
}

int main() {
    cv::Mat img = cv::imread("resim.jpg");
    if (img.empty()) {
        std::cout << "Goruntu bulunamadi! 'goruntu.jpg' dosyasini kontrol edin." << std::endl;
        return -1;
    }

    // Görüntüyü Mavi, Yeşil ve Kırmızı kanallarına ayır
    std::vector<cv::Mat> bgr_planes;
    cv::split(img, bgr_planes);

    // Hesaplanan değerleri tutacağımız vektörler
    std::vector<int> b_hist, g_hist, r_hist;

    // Kendi fonksiyonumuzu her kanal için ayrı ayrı çalıştırıyoruz
    customCalcHist(bgr_planes[0], b_hist);
    customCalcHist(bgr_planes[1], g_hist);
    customCalcHist(bgr_planes[2], r_hist);

    // Çizim sırasında grafiğin ekrana sığması (Normalizasyon) için en çok tekrar eden piksel sayısını buluyoruz
    int max_val = 0;
    for (int i = 0; i < 256; i++) {
        max_val = std::max({max_val, b_hist[i], g_hist[i], r_hist[i]});
    }

    // Çizim için siyah pencere oluştur
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / 256);
    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

    // Kendi hesapladığımız değerleri grafiğe çiz
    for (int i = 1; i < 256; i++) {
        // Değerleri pencere yüksekliğine sığacak şekilde oranla
        int b_y1 = hist_h - cvRound(((double)b_hist[i - 1] / max_val) * hist_h);
        int b_y2 = hist_h - cvRound(((double)b_hist[i] / max_val) * hist_h);
        
        int g_y1 = hist_h - cvRound(((double)g_hist[i - 1] / max_val) * hist_h);
        int g_y2 = hist_h - cvRound(((double)g_hist[i] / max_val) * hist_h);
        
        int r_y1 = hist_h - cvRound(((double)r_hist[i - 1] / max_val) * hist_h);
        int r_y2 = hist_h - cvRound(((double)r_hist[i] / max_val) * hist_h);

        // Çizgileri çek
        cv::line(histImage, cv::Point(bin_w * (i - 1), b_y1), cv::Point(bin_w * i, b_y2), cv::Scalar(255, 0, 0), 2, 8, 0);
        cv::line(histImage, cv::Point(bin_w * (i - 1), g_y1), cv::Point(bin_w * i, g_y2), cv::Scalar(0, 255, 0), 2, 8, 0);
        cv::line(histImage, cv::Point(bin_w * (i - 1), r_y1), cv::Point(bin_w * i, r_y2), cv::Scalar(0, 0, 255), 2, 8, 0);
    }

    // Sonucu diske kaydet
    cv::imwrite("ozel_histogram_sonuc.jpg", histImage);
    std::cout << "Kendi fonksiyonumuz calisti! Histogram 'ozel_histogram_sonuc.jpg' olarak kaydedildi." << std::endl;

    return 0;
}
