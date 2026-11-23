#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    Mat left = imread("scene1.row3.col1.ppm", IMREAD_GRAYSCALE);
    Mat right = imread("scene1.row3.col3.ppm", IMREAD_GRAYSCALE);

    if (left.empty() || right.empty()) {
        cerr << "Görüntüler yüklenemedi! Dosya yollarını kontrol et." << endl;
        return -1;
    }

    int numDisparities = 32;
    int blockSize = 15;
    Ptr<StereoBM> stereo = StereoBM::create(numDisparities, blockSize);

    Mat disp16S, disp8U, dispColor;
    stereo->compute(left, right, disp16S);

    normalize(disp16S, disp8U, 0, 255, NORM_MINMAX, CV_8U);
    applyColorMap(disp8U, dispColor, COLORMAP_JET);

    imwrite("disparity_gray.png", disp8U);
    imwrite("disparity_color.png", dispColor);

    cout << "Haritalar basariyla kaydedildi!" << endl;
    return 0;
}
