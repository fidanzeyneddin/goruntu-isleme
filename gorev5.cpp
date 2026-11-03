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

    vector<double> cos_t(360), sin_t(360);
    for (int t = 0; t < 360; t++) {
        cos_t[t] = cos(t * CV_PI / 180.0);
        sin_t[t] = sin(t * CV_PI / 180.0);
    }

    vector<vector<vector<int>>> acc3D(width, vector<vector<int>>(height, vector<int>(max_r - min_r + 1, 0)));
    
    // Sadece merkezleri takip eden 2D oylama matrisi
    vector<vector<int>> concentricCenters(width, vector<int>(height, 0));
    int threshold = 60; // Tek bir r için olgunluk eşiği

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (edges.at<uchar>(y, x) > 0) {
                for (int r = min_r; r <= max_r; r++) {
                    for (int t = 0; t < 360; t += 5) {
                        int a = round(x - r * cos_t[t]);
                        int b = round(y - r * sin_t[t]);

                        if (a >= 0 && a < width && b >= 0 && b < height) {
                            acc3D[a][b][r - min_r]++;
                            
                            // Oylama anında eş merkezlilik kontrolü: 
                            // Eğer o anki oy sayısı TAM olarak eşiğe ulaştıysa, o merkeze +1 "çember buldun" oyu ver.
                            if (acc3D[a][b][r - min_r] == threshold) {
                                concentricCenters[a][b]++; 
                            }
                        }
                    }
                }
            }
        }
    }

    Mat result = img.clone();
    
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            // Sadece aynı merkezde 1'den fazla çember olgunlaştıysa çiz
            if (concentricCenters[x][y] > 1) { 
                for (int r = min_r; r <= max_r; r++) {
                    if (acc3D[x][y][r - min_r] >= threshold) {
                        circle(result, Point(x, y), r, Scalar(0, 165, 255), 2); // Turuncu
                    }
                }
            }
        }
    }

    imwrite("gorev5_esmerkezli_cemberler.jpg", result);
    cout << "Görev 5 tamamlandi." << endl;
    return 0;
}
