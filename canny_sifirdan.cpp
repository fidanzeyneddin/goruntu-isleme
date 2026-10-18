#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat applyGaussian(const Mat& src) {
    Mat dst;
    GaussianBlur(src, dst, Size(5, 5), 1.4);
    return dst;
}

void computeGradients(const Mat& src, Mat& magnitude, Mat& angle) {
    Mat gx, gy;
    Sobel(src, gx, CV_32F, 1, 0, 3);
    Sobel(src, gy, CV_32F, 0, 1, 3);
    
    cartToPolar(gx, gy, magnitude, angle, true); 
}

Mat nonMaxSuppression(const Mat& magnitude, const Mat& angle) {
    Mat nms = Mat::zeros(magnitude.size(), CV_32F);
    
    for (int i = 1; i < magnitude.rows - 1; i++) {
        for (int j = 1; j < magnitude.cols - 1; j++) {
            float a = angle.at<float>(i, j);
            float mag = magnitude.at<float>(i, j);
            float q = 255, r = 255;

            if ((a >= 0 && a < 22.5) || (a >= 157.5 && a <= 180)) {
                q = magnitude.at<float>(i, j+1); 
                r = magnitude.at<float>(i, j-1);
            } else if (a >= 22.5 && a < 67.5) {
                q = magnitude.at<float>(i+1, j-1); 
                r = magnitude.at<float>(i-1, j+1);
            } else if (a >= 67.5 && a < 112.5) {
                q = magnitude.at<float>(i+1, j); 
                r = magnitude.at<float>(i-1, j);
            } else if (a >= 112.5 && a < 157.5) {
                q = magnitude.at<float>(i-1, j-1); 
                r = magnitude.at<float>(i+1, j+1);
            }

            if (mag >= q && mag >= r) {
                nms.at<float>(i, j) = mag;
            } else {
                nms.at<float>(i, j) = 0;
            }
        }
    }
    return nms;
}

Mat hysteresisThresholding(const Mat& nms, float lowThresh, float highThresh) {
    Mat result = Mat::zeros(nms.size(), CV_8U);
    
    for (int i = 1; i < nms.rows - 1; i++) {
        for (int j = 1; j < nms.cols - 1; j++) {
            float mag = nms.at<float>(i, j);
            if (mag >= highThresh) {
                result.at<uchar>(i, j) = 255; 
            } else if (mag >= lowThresh) {
                result.at<uchar>(i, j) = 50;  
            }
        }
    }
    
    for (int i = 1; i < result.rows - 1; i++) {
        for (int j = 1; j < result.cols - 1; j++) {
            if (result.at<uchar>(i, j) == 50) {
                if (result.at<uchar>(i+1, j) == 255 || result.at<uchar>(i-1, j) == 255 ||
                    result.at<uchar>(i, j+1) == 255 || result.at<uchar>(i, j-1) == 255 ||
                    result.at<uchar>(i+1, j+1) == 255 || result.at<uchar>(i-1, j-1) == 255 ||
                    result.at<uchar>(i-1, j+1) == 255 || result.at<uchar>(i+1, j-1) == 255) {
                    
                    result.at<uchar>(i, j) = 255; 
                } else {
                    result.at<uchar>(i, j) = 0;   
                }
            }
        }
    }
    return result;
}

int main() {
    Mat img = imread("resim.jpg", IMREAD_GRAYSCALE);
    if (img.empty()) {
        cout << "Hata: resim.jpg bulunamadi!" << endl;
        return -1;
    }

    Mat blurred = applyGaussian(img);
    
    Mat magnitude, angle;
    computeGradients(blurred, magnitude, angle);
    
    Mat nms = nonMaxSuppression(magnitude, angle);
  
    Mat finalEdges = hysteresisThresholding(nms, 20.0, 60.0); 

    imwrite("canny_sifirdan_sonuc.jpg", finalEdges);
    cout << "Manuel Canny algoritmasi tamamlandi. 'canny_sifirdan_sonuc.jpg' kaydedildi." << endl;

    return 0;
}
