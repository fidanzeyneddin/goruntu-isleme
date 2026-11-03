#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>

using namespace cv;
using namespace std;

int main() {
    Mat img = imread("image.jpg");
    if (img.empty()) return -1;

    Mat gray, edges;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Canny(gray, edges, 50, 150);

    int width = edges.cols;
    int height = edges.rows;
    int min_r = 20, max_r = 100;
    
    // 1. Sin ve Cos ön hesabı
    vector<double> cos_t(360), sin_t(360);
    for (int t = 0; t < 360; t++) {
        cos_t[t] = cos(t * CV_PI / 180.0);
        sin_t[t] = sin(t * CV_PI / 180.0);
    }

    // 2. 3-Boyutlu Oylama Matrisi [x][y][r]
    vector<vector<vector<int>>> acc3D(width, vector<vector<int>>(height, vector<int>(max_r - min_r + 1, 0)));

    // 3. Oylama (5 derecelik adımlarla, tam çember için max oy 72 olur)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (edges.at<uchar>(y, x) > 0) {
                for (int r = min_r; r <= max_r; r++) {
                    for (int t = 0; t < 360; t+=5) { 
                        int a = round(x - r * cos_t[t]);
                        int b = round(y - r * sin_t[t]);
                        if (a >= 0 && a < width && b >= 0 && b < height) {
                            acc3D[a][b][r - min_r]++;
                        }
                    }
                }
            }
        }
    }

    // 4. Çizim (Eşik değeri 60 / Max 72 üzerinden)
    Mat result = img.clone();
    int threshold = 60; 

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            for (int r = min_r; r <= max_r; r++) {
                if (acc3D[x][y][r - min_r] >= threshold) {
                    circle(result, Point(x, y), r, Scalar(0, 255, 255), 2); // Sarı
                }
            }
        }
    }

    imwrite("gorev4_kendi_cemberlerimiz.jpg", result);
    cout << "Görev 4 tamamlandi." << endl;
    return 0;
}
