#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main() {
    VideoCapture cap("test.mp4"); // Colab'a yüklediğin video dosyası
    if (!cap.isOpened()) {
        cout << "Video acilamadi! Dosya adini kontrol et." << endl;
        return -1;
    }

    int genislik = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int yukseklik = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(CAP_PROP_FPS);
    if (fps <= 0) fps = 30.0;

    // Çıktıyı video olarak kaydetmek için VideoWriter
    VideoWriter writer("sonuc_video.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(genislik, yukseklik));

    Mat oncekiKare, oncekiGri;
    cap >> oncekiKare;
    if (oncekiKare.empty()) return -1;

    cvtColor(oncekiKare, oncekiGri, COLOR_BGR2GRAY);
    GaussianBlur(oncekiGri, oncekiGri, Size(21, 21), 0);

    Mat mevcutKare, mevcutGri, fark, esiklenmis;

    while (true) {
        cap >> mevcutKare;
        if (mevcutKare.empty()) break;

        cvtColor(mevcutKare, mevcutGri, COLOR_BGR2GRAY);
        GaussianBlur(mevcutGri, mevcutGri, Size(21, 21), 0);

        absdiff(oncekiGri, mevcutGri, fark);
        threshold(fark, esiklenmis, 25, 255, THRESH_BINARY);
        dilate(esiklenmis, esiklenmis, Mat(), Point(-1, -1), 2);

        if (countNonZero(esiklenmis) > 1500) {
            vector<vector<Point>> konturlar;
            findContours(esiklenmis, konturlar, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

            double maxAlan = 0;
            Rect enBuyukROI;
            for (const auto& kontur : konturlar) {
                double alan = contourArea(kontur);
                if (alan > maxAlan) {
                    maxAlan = alan;
                    enBuyukROI = boundingRect(kontur);
                }
            }

            if (maxAlan > 500) {
                Mat roi = mevcutKare(enBuyukROI);
                Mat roiYCrCb, tenMaskesi;

                cvtColor(roi, roiYCrCb, COLOR_BGR2YCrCb);
                inRange(roiYCrCb, Scalar(0, 133, 77), Scalar(255, 173, 127), tenMaskesi);

                roi.setTo(Scalar(0, 255, 0), tenMaskesi);
                rectangle(mevcutKare, enBuyukROI, Scalar(255, 0, 0), 2);
            }
        }

        writer.write(mevcutKare);
        oncekiGri = mevcutGri.clone();
    }

    cap.release();
    writer.release();
    cout << "Islem tamam! sonuc_video.mp4 olusturuldu." << endl;
    return 0;
}
