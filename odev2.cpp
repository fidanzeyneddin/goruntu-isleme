#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

void cemberBulVeCiz(const Mat& maske, Mat& sonucResmi, const string& ciktiAdi) {
    // 1. Maskede hiç kırmızı yakalanmış mı kontrol et
    int beyazPiksel = countNonZero(maske);
    cout << ciktiAdi << " maskesindeki beyaz piksel sayisi: " << beyazPiksel << endl;

    Mat temizMaske;
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));

    morphologyEx(maske, temizMaske, MORPH_CLOSE, kernel);
    GaussianBlur(temizMaske, temizMaske, Size(5, 5), 1.5, 1.5);

    imwrite(ciktiAdi + "_maske.jpg", temizMaske);

    vector<Vec3f> cemberler;
    // dp=1.2 hafif elips/bozuk çemberlere tolerans tanır, minRadius=10 küçük levhaları da yakalar
    HoughCircles(temizMaske, cemberler, HOUGH_GRADIENT, 1.2,
                 temizMaske.rows / 6, // Merkezler arası min mesafe
                 50, 15,              // param1 (Canny), param2 (Oylama eşiği - esnek)
                 10, 180);            // minRadius, maxRadius

    for (size_t i = 0; i < cemberler.size(); i++) {
        Point merkez(cvRound(cemberler[i][0]), cvRound(cemberler[i][1]));
        int yaricap = cvRound(cemberler[i][2]);
        circle(sonucResmi, merkez, yaricap, Scalar(0, 255, 0), 3);
        circle(sonucResmi, merkez, 3, Scalar(0, 0, 255), -1);
    }

    imwrite(ciktiAdi + ".jpg", sonucResmi);
    cout << ciktiAdi << " icin bulunan cember sayisi: " << cemberler.size() << endl;
}

int main() {
    Mat hamImg = imread("indir (9).png");
    if (hamImg.empty()) {
        cout << "Hata: trafik_isareti.jpg bulunamadi!" << endl;
        return -1;
    }

    cout << "Orijinal boyut: " << hamImg.cols << "x" << hamImg.rows << endl;

    // En-boy oranını BOZMADAN uzun kenarı 400 piksele ölçekle
    Mat img;
    double olcek = 400.0 / max(hamImg.cols, hamImg.rows);
    resize(hamImg, img, Size(), olcek, olcek);
    copyMakeBorder(img, img, 30, 30, 30, 30, BORDER_CONSTANT, Scalar(0, 0, 0));

    // --- YÖNTEM 1: HSV ---
    Mat hsv, maske1, maske2, hsvKirmiziMaske;
    cvtColor(img, hsv, COLOR_BGR2HSV);
    inRange(hsv, Scalar(0, 40, 30), Scalar(15, 255, 255), maske1);
    inRange(hsv, Scalar(160, 40, 30), Scalar(180, 255, 255), maske2);
    bitwise_or(maske1, maske2, hsvKirmiziMaske);

    Mat sonucHSV = img.clone();
    cemberBulVeCiz(hsvKirmiziMaske, sonucHSV, "sonuc_hsv");

    // --- YÖNTEM 2: YCrCb ---
    Mat ycrcb, ycrcbKirmiziMaske;
    cvtColor(img, ycrcb, COLOR_BGR2YCrCb);
    inRange(ycrcb, Scalar(0, 140, 0), Scalar(255, 255, 140), ycrcbKirmiziMaske);

    Mat sonucYCrCb = img.clone();
    cemberBulVeCiz(ycrcbKirmiziMaske, sonucYCrCb, "sonuc_ycrcb");

    return 0;
}
