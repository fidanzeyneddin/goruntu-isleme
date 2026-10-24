#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    cv::Mat src = cv::imread("resim.jpg", cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cerr << "resim.jpg bulunamadi!" << std::endl;
        return -1;
    }

    std::vector<cv::Mat> bgr_planes;
    cv::split(src, bgr_planes);

    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange[] = { range };

    cv::Mat b_hist, g_hist, r_hist;
    cv::calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, histRange);
    cv::calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, histRange);
    cv::calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, histRange);

    std::cout << "OpenCV ile B, G, R kanallari icin histogramlar hesaplandi." << std::endl;
    return 0;
}
