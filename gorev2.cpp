#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>

using namespace cv;
using namespace std;

int main() {
    Mat img = imread("image.jpg");
    if (img.empty()) return -1;

    Mat gray, edges, median_edges;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Canny(gray, edges, 50, 150);
    medianBlur(edges, median_edges, 5);

    int width = edges.cols;
    int height = edges.rows;
    int max_rho = ceil(sqrt(width * width + height * height));

    // 1. Sin ve Cos değerlerini önceden hesapla
    vector<double> cos_t(180), sin_t(180);
    for (int t = 0; t < 180; t++) {
        double theta = t * CV_PI / 180.0;
        cos_t[t] = cos(theta);
        sin_t[t] = sin(theta);
    }

    // 2. Oylama (Accumulator) matrisi: Boyut(2*max_rho, 180)
    Mat accumulator = Mat::zeros(2 * max_rho, 180, CV_32SC1);

    // 3. Oylama İşlemi
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (median_edges.at<uchar>(y, x) > 0) {
                for (int t = 0; t < 180; t++) {
                    int rho = round(x * cos_t[t] + y * sin_t[t]);
                    accumulator.at<int>(rho + max_rho, t)++;
                }
            }
        }
    }

    // Oylama matrisini görselleştirmek için kaydet
    Mat acc_img;
    normalize(accumulator, acc_img, 0, 255, NORM_MINMAX, CV_8UC1);
    imwrite("gorev2_oylama_matrisi.jpg", acc_img);

    // 4. Eşik değere göre doğruları çiz (Test resmindeki çizgilerimiz 400px uzunluğunda, eşiği 200 yapıyoruz)
    int threshold_votes = 200;
    Mat result = img.clone();

    for (int r = 0; r < 2 * max_rho; r++) {
        for (int t = 0; t < 180; t++) {
            if (accumulator.at<int>(r, t) >= threshold_votes) {
                int rho = r - max_rho;
                double a = cos_t[t], b = sin_t[t];
                double x0 = a * rho, y0 = b * rho;
                Point pt1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
                Point pt2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));
                line(result, pt1, pt2, Scalar(0, 255, 0), 2, LINE_AA); // Yeşil
            }
        }
    }

    imwrite("gorev2_kendi_dogrularimiz.jpg", result);
    cout << "Görev 2 tamamlandi." << endl;
    return 0;
}
