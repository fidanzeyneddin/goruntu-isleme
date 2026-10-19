#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main() {
    // 1. Resmi renkli olarak oku (IMREAD_COLOR varsayılandır, BGR formatında okur)
    Mat img = imread("resim.jpg", IMREAD_COLOR);
    if (img.empty()) {
        cout << "Hata: resim.jpg bulunamadi!" << endl;
        return -1;
    }

    // 2. Görüntüyü 3 ayrı kanala (Mavi, Yeşil, Kırmızı) bölme
    vector<Mat> bgr_planes;
    split(img, bgr_planes);

    // 3. Histogram ayarları
    int histSize = 256; // 0-255 arası piksel değerleri
    float range[] = { 0, 256 }; // Değer aralığı
    const float* histRange[] = { range };
    bool uniform = true, accumulate = false;

    // Her kanal için histogram matrislerini oluşturma
    Mat b_hist, g_hist, r_hist;

    // 4. Histogramları hesaplama
    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, histRange, uniform, accumulate); // Mavi
    calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, histRange, uniform, accumulate); // Yeşil
    calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, histRange, uniform, accumulate); // Kırmızı

    // 5. Histogramları çizmek için siyah bir tuval (resim) oluşturma
    int hist_w = 512, hist_h = 400; // Genişlik ve yükseklik
    int bin_w = cvRound((double)hist_w / histSize);
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

    // 6. Histogram değerlerini tuvalin yüksekliğine sığacak şekilde normalize etme (0 ile 400 arasına sıkıştırma)
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    // 7. Her bir piksel değeri (0-255) için grafiğe çizgi (line) çizme
    for (int i = 1; i < histSize; i++) {
        // Mavi kanal için çizgi
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
                        Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
                        Scalar(255, 0, 0), 2, 8, 0);
        // Yeşil kanal için çizgi
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
                        Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
                        Scalar(0, 255, 0), 2, 8, 0);
        // Kırmızı kanal için çizgi
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
                        Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
                        Scalar(0, 0, 255), 2, 8, 0);
    }

    // 8. Çizilen grafiği kaydetme
    imwrite("renkli_histogram_grafik.jpg", histImage);
    cout << "Islem basarili! BGR histogram grafigi 'renkli_histogram_grafik.jpg' olarak kaydedildi." << endl;

    return 0;
}
