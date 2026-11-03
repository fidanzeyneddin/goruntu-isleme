#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    Mat img = imread("image.jpg");
    if (img.empty()) return -1;

    Mat gray, edges, median_edges;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Canny(gray, edges, 50, 150);
    medianBlur(edges, median_edges, 5);

    vector<Vec3f> circles;
    HoughCircles(median_edges, circles, HOUGH_GRADIENT, 1, 
                 20,  // Çember merkezleri arası minimum mesafe
                 100, // Canny üst eşik
                 30,  // Merkez bulma eşiği
                 20, 100); // Min ve Max yarıçap

    Mat result = img.clone();
    for (size_t i = 0; i < circles.size(); i++) {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        circle(result, center, 3, Scalar(0, 0, 255), -1, 8, 0); // Merkez (Kırmızı)
        circle(result, center, radius, Scalar(255, 0, 255), 3, 8, 0); // Çevre (Mor)
    }

    imwrite("gorev3_opencv_cemberler.jpg", result);
    cout << "Görev 3 tamamlandi." << endl;
    return 0;
}
