#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <climits>
#include <cmath>

using namespace cv;
using namespace std;
using namespace std::chrono;

int calcSAD(const Mat& left, const Mat& right, int x, int y, int d, int halfW) {
    int sad = 0;
    for (int dy = -halfW; dy <= halfW; dy++) {
        for (int dx = -halfW; dx <= halfW; dx++) {
            int pL = left.at<uchar>(y + dy, x + dx);
            int pR = right.at<uchar>(y + dy, x - d + dx);
            sad += abs(pL - pR);
        }
    }
    return sad;
}

Mat denseBlockMatching(const Mat& left, const Mat& right, int blockSize, int maxDisp) {
    int halfW = blockSize / 2;
    Mat disp = Mat::zeros(left.size(), CV_8UC1);

    for (int y = halfW; y < left.rows - halfW; y++) {
        for (int x = halfW; x < left.cols - halfW; x++) {
            int bestCost = INT_MAX, bestD = 0;
            for (int d = 0; d < maxDisp; d++) {
                if (x - d < halfW) break;
                int cost = calcSAD(left, right, x, y, d, halfW);
                if (cost < bestCost) {
                    bestCost = cost;
                    bestD = d;
                }
            }
            disp.at<uchar>(y, x) = static_cast<uchar>(bestD);
        }
    }
    return disp;
}

Mat sparseBlockMatching(const Mat& left, const Mat& right,
                        const Mat& edgesL, const Mat& edgesR_dilated,
                        int blockSize, int maxDisp) {
    int halfW = blockSize / 2;
    Mat disp = Mat::zeros(left.size(), CV_8UC1);

    for (int y = halfW; y < left.rows - halfW; y++) {
        for (int x = halfW; x < left.cols - halfW; x++) {
            if (edgesL.at<uchar>(y, x) == 0) continue;

            int bestCost = INT_MAX, bestD = 0;
            bool found = false;
            for (int d = 0; d < maxDisp; d++) {
                if (x - d < halfW) break;
                if (edgesR_dilated.at<uchar>(y, x - d) == 0) continue;

                int cost = calcSAD(left, right, x, y, d, halfW);
                if (cost < bestCost) {
                    bestCost = cost;
                    bestD = d;
                    found = true;
                }
            }
            if (found) disp.at<uchar>(y, x) = static_cast<uchar>(bestD);
        }
    }
    return disp;
}

// Doğruluk ölçümü (Referans: SGBM)
void evaluateAccuracy(const string& label, const Mat& testDisp, const Mat& refSGBM,
                      const Mat& validMask, int borderX) {
    double sumAbsErr = 0.0;
    int badPixels = 0; // Hata > 1.0 piksel olanlar
    int count = 0;

    for (int y = 0; y < testDisp.rows; y++) {
        for (int x = borderX; x < testDisp.cols; x++) {
            float refVal = refSGBM.at<float>(y, x);
            if (refVal <= 0.0f) continue; // SGBM'de geçersiz (occlusion) pikselleri atla
            if (!validMask.empty() && validMask.at<uchar>(y, x) == 0) continue;

            float testVal = static_cast<float>(testDisp.at<uchar>(y, x));
            float err = fabs(testVal - refVal);
            sumAbsErr += err;
            if (err > 1.0f) badPixels++;
            count++;
        }
    }

    double mae = (count > 0) ? (sumAbsErr / count) : 0.0;
    double badPct = (count > 0) ? (100.0 * badPixels / count) : 0.0;

    cout << left << setw(34) << label
         << " | Piksel: " << setw(6) << count
         << " | MAE: " << fixed << setprecision(3) << setw(6) << mae << " px"
         << " | Hatali (>1px): %" << setprecision(2) << badPct << endl;
}

int main() {
    Mat leftImg  = imread("tsukuba/scene1.row3.col1.ppm", IMREAD_GRAYSCALE);
    Mat rightImg = imread("tsukuba/scene1.row3.col3.ppm", IMREAD_GRAYSCALE);

    int blockSize = 7;
    int maxDisp = 32; // 1. ve 3. resim arası ~28 px kayma olduğu için 32 seçildi

    // 1. OpenCV StereoSGBM (Referans Yoğun Yöntem)
    auto tS1 = high_resolution_clock::now();
    Ptr<StereoSGBM> sgbm = StereoSGBM::create(
        0, maxDisp, blockSize,
        8 * blockSize * blockSize,
        32 * blockSize * blockSize
    );
    Mat sgbm16S, sgbmFloat;
    sgbm->compute(leftImg, rightImg, sgbm16S);
    sgbm16S.convertTo(sgbmFloat, CV_32F, 1.0 / 16.0);
    auto tS2 = high_resolution_clock::now();
    double sgbmMs = duration<double, milli>(tS2 - tS1).count();

    // 2. Kendi Dense SAD yöntemimiz
    auto tD1 = high_resolution_clock::now();
    Mat dispDense = denseBlockMatching(leftImg, rightImg, blockSize, maxDisp);
    auto tD2 = high_resolution_clock::now();
    double denseMs = duration<double, milli>(tD2 - tD1).count();

    // 3. Kendi Sparse (Canny + Dilate + SAD) yöntemimiz
    auto tSp1 = high_resolution_clock::now();
    Mat blurL, blurR, edgesL, edgesR, edgesR_dilated;
    GaussianBlur(leftImg, blurL, Size(3, 3), 0);
    GaussianBlur(rightImg, blurR, Size(3, 3), 0);
    Canny(blurL, edgesL, 50, 150);
    Canny(blurR, edgesR, 50, 150);
    dilate(edgesR, edgesR_dilated, getStructuringElement(MORPH_RECT, Size(3, 3)));
    Mat dispSparse = sparseBlockMatching(leftImg, rightImg, edgesL, edgesR_dilated, blockSize, maxDisp);
    auto tSp2 = high_resolution_clock::now();
    double sparseMs = duration<double, milli>(tSp2 - tSp1).count();

    cout << "=== HIZ KIYASLAMASI (maxDisp=32, 7x7) ===" << endl;
    cout << "OpenCV StereoSGBM : " << fixed << setprecision(2) << sgbmMs << " ms" << endl;
    cout << "Dense 7x7 SAD     : " << denseMs << " ms" << endl;
    cout << "Sparse Canny+SAD  : " << sparseMs << " ms" << endl;
    cout << "Hizlanma (Dense/Sparse): " << (denseMs / sparseMs) << "x" << endl << endl;

    cout << "=== DOGRULUK KIYASLAMASI (Referans: OpenCV StereoSGBM) ===" << endl;
    Mat sparseValidMask = (dispSparse > 0); // Sadece sparse'ın eşleştirebildiği kenarlar
    evaluateAccuracy("Dense SAD (Tum Goruntu)",     dispDense,  sgbmFloat, Mat(),           maxDisp);
    evaluateAccuracy("Dense SAD (Sadece Kenarlar)", dispDense,  sgbmFloat, sparseValidMask, maxDisp);
    evaluateAccuracy("Sparse SAD (Sadece Kenarlar)",dispSparse, sgbmFloat, sparseValidMask, maxDisp);

    // Görselleştirme için kaydet
    Mat sgbm8U;
    sgbmFloat.convertTo(sgbm8U, CV_8UC1);
    imwrite("final_sgbm.png", sgbm8U);
    imwrite("final_dense.png", dispDense);
    imwrite("final_sparse.png", dispSparse);

    return 0;
}
