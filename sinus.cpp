#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

void generateSineWave(std::vector<double>& vec, double amplitude, double period, double phase) {
    for (size_t t = 0; t < vec.size(); ++t) {
        vec[t] = amplitude * std::sin((2.0 * CV_PI * t / period) + phase);
    }
}

int main() {
    int size = 512; 
    std::vector<double> wave(size);
    
    double amplitude = 100.0;
    double period = 128.0;
    double phase = 0.0;
    
    generateSineWave(wave, amplitude, period, phase);
    
    int height = 300;
    int width = size;
    cv::Mat plot = cv::Mat::zeros(height, width, CV_8UC3);
    
    int centerY = height / 2;
    
    cv::line(plot, cv::Point(0, centerY), cv::Point(width, centerY), cv::Scalar(100, 100, 100), 1);
    
    for (int i = 0; i < size - 1; ++i) {
        int y1 = centerY - static_cast<int>(wave[i]); 
        int y2 = centerY - static_cast<int>(wave[i+1]);
        cv::line(plot, cv::Point(i, y1), cv::Point(i+1, y2), cv::Scalar(0, 255, 0), 2);
    }
    
    // Colab'de imshow çalışmaz, bu yüzden resmi dosyaya kaydediyoruz.
    cv::imwrite("sinus_dalgasi.png", plot);
    std::cout << "Gorsel basariyla kaydedildi." << std::endl;
    
    return 0;
}
