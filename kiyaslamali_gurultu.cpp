#include <iostream>
#include <cmath>
#include <random>
#include <opencv2/opencv.hpp>

// 1. Kırpma Fonksiyonları (2'nin kuvveti için)
int getClosestPowerOf2(int n) { return 1 << static_cast<int>(std::floor(std::log2(n))); }

cv::Mat cropToPowerOf2(const cv::Mat& img) {
    int new_width = getClosestPowerOf2(img.cols);
    int new_height = getClosestPowerOf2(img.rows);
    return img(cv::Rect(0, 0, new_width, new_height)).clone();
}

// 2. Gauss Gürültüsü (Parametreler: Ortalama ve Standart Sapma/Kuvvet)
void addGaussianNoise(const cv::Mat& src, cv::Mat& dst, double mean, double stddev) {
    cv::Mat noise(src.size(), CV_32F);
    cv::randn(noise, mean, stddev);
    cv::Mat src_float; 
    src.convertTo(src_float, CV_32F);
    cv::Mat dst_float = src_float + noise;
    dst_float.convertTo(dst, CV_8U);
}

// 3. Salt & Pepper Gürültüsü (Parametre: Etkilenecek Piksel Oranı)
void addSaltAndPepperNoise(const cv::Mat& src, cv::Mat& dst, double noise_ratio) {
    dst = src.clone();
    int noise_pixels = static_cast<int>(src.rows * src.cols * noise_ratio);
    std::random_device rd; 
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> disX(0, src.cols - 1);
    std::uniform_int_distribution<> disY(0, src.rows - 1);
    for (int i = 0; i < noise_pixels; ++i) {
        dst.at<uchar>(disY(gen), disX(gen)) = (i % 2 == 0) ? 0 : 255;
    }
}

// 4. 2D DFT Harmonik Matrisi Hesaplama
cv::Mat compute2D_DFT(const cv::Mat& src) {
    cv::Mat padded; 
    src.convertTo(padded, CV_32F);
    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexI; 
    cv::merge(planes, 2, complexI);
    cv::dft(complexI, complexI);
    cv::split(complexI, planes);
    cv::Mat magI; 
    cv::magnitude(planes[0], planes[1], magI);
    return magI;
}

// 5. Spektrumu Renklendirip Görselleştirme (Merkez Kaydırma ile)
// DİKKAT: cv::Mat magI yerine cv::Mat& magI idi, derleme hatasını önlemek için pass-by-value yapıyoruz 
// veya fonksiyon içinde referans alıp değiştirmemek için `cv::Mat processColoredSpectrum(cv::Mat magI)`
cv::Mat processColoredSpectrum(cv::Mat magI) {
    magI += cv::Scalar::all(1);
    cv::log(magI, magI);
    magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));
    int cx = magI.cols / 2, cy = magI.rows / 2;
    cv::Mat q0(magI, cv::Rect(0, 0, cx, cy)), q1(magI, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(magI, cv::Rect(0, cy, cx, cy)), q3(magI, cv::Rect(cx, cy, cx, cy)), tmp;
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);
    
    cv::normalize(magI, magI, 0, 1, cv::NORM_MINMAX);
    cv::Mat spectrum8U; 
    magI.convertTo(spectrum8U, CV_8U, 255.0);
    cv::Mat coloredSpectrum;
    cv::applyColorMap(spectrum8U, coloredSpectrum, cv::COLORMAP_JET);
    return coloredSpectrum;
}

int main() {
    // "orman_web_2.jpg" (Tuğla duvar) üzerinden işlem yapılıyor
    cv::Mat original = cv::imread("orman_web.jpg", cv::IMREAD_GRAYSCALE);
    if(original.empty()) return -1;
    cv::Mat cropped = cropToPowerOf2(original);

    // 1. Orijinal Görüntü ve Spektrumu
    // DÜZELTME: processColoredSpectrum artık argümanı değer (value) olarak aldığı için bu şekilde çalışır.
    cv::Mat orig_mag = compute2D_DFT(cropped);
    cv::Mat orig_dft = processColoredSpectrum(orig_mag);
    
    cv::Mat orig_bgr; cv::cvtColor(cropped, orig_bgr, cv::COLOR_GRAY2BGR);
    cv::Mat c_orig; cv::hconcat(orig_bgr, orig_dft, c_orig);
    cv::imwrite("1_orijinal_kiyas.jpg", c_orig);

    // 2. Gauss Gürültüsü ve Spektrumu (stddev = 50.0)
    cv::Mat img_gauss; 
    addGaussianNoise(cropped, img_gauss, 0.0, 50.0);
    
    cv::Mat gauss_mag = compute2D_DFT(img_gauss);
    cv::Mat gauss_dft = processColoredSpectrum(gauss_mag);
    
    cv::Mat gauss_bgr; cv::cvtColor(img_gauss, gauss_bgr, cv::COLOR_GRAY2BGR);
    cv::Mat c_gauss; cv::hconcat(gauss_bgr, gauss_dft, c_gauss);
    cv::imwrite("2_gauss_kiyas.jpg", c_gauss);

    // 3. Salt & Pepper Gürültüsü ve Spektrumu (%15 bozulma)
    cv::Mat img_sp; 
    addSaltAndPepperNoise(cropped, img_sp, 0.15);
    
    cv::Mat sp_mag = compute2D_DFT(img_sp);
    cv::Mat sp_dft = processColoredSpectrum(sp_mag);
    
    cv::Mat sp_bgr; cv::cvtColor(img_sp, sp_bgr, cv::COLOR_GRAY2BGR);
    cv::Mat c_sp; cv::hconcat(sp_bgr, sp_dft, c_sp);
    cv::imwrite("3_sp_kiyas.jpg", c_sp);

    std::cout << "Kiyaslama gorselleri basariyla uretildi." << std::endl;
    return 0;
}
