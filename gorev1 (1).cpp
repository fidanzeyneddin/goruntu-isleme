#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    Mat img = imread("image.jpg");
    if (img.empty()) {
        cout << "Resim bulunamadi!" << endl;
        return -1;
    }

    Mat gray, edges, median_edges;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // 1. Canny ile kenar tespiti
    Canny(gray, edges, 50, 150);

    // 2. Median filtre uygulama
    medianBlur(edges, median_edges, 5);

    Mat result_standard = img.clone();
    Mat result_probabilistic = img.clone();

    // 3. Standart Hough Line Transform (Kırmızı çizgiler)
    vector<Vec2f> lines;
    HoughLines(median_edges, lines, 1, CV_PI / 180, 100); // Eşik değeri 100
    for (size_t i = 0; i < lines.size(); i++) {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        line(result_standard, pt1, pt2, Scalar(0, 0, 255), 2, LINE_AA);
    }

    // 4. Probabilistic Hough Line Transform (Mavi çizgiler)
    vector<Vec4i> linesP;
    HoughLinesP(median_edges, linesP, 1, CV_PI / 180, 50, 50, 10);
    for (size_t i = 0; i < linesP.size(); i++) {
        Vec4i l = linesP[i];
        line(result_probabilistic, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 0, 0), 2, LINE_AA);
    }

    imwrite("gorev1_standart_hough.jpg", result_standard);
    imwrite("gorev1_probabilistic_hough.jpg", result_probabilistic);
    cout << "Görev 1 tamamlandi. Görüntüler kaydedildi." << endl;
    
    return 0;
}
