#include <iostream>
#include <cmath>
#include <random>
#include <opencv2/opencv.hpp>

// 1. Kırpma Fonksiyonu
int getClosestPowerOf2(int n) { return 1 << static_cast<int>(std::floor(std::log2(n))); }

cv::Mat cropToPowerOf2(const cv::Mat& img) {
    int new_width = getClosestPowerOf2(img.cols);
    int new_height = getClosestPowerOf2(img.rows);
    return img(cv::Rect(0, 0, new_width, new_height)).clone();
}

// 2. Gürültü Ekleme Fonksiyonları
void addGaussianNoise(const cv::Mat& src, cv::Mat& dst, double mean, double stddev) {
    cv::Mat noise(src.size(), CV_32F);
    cv::randn(noise, mean, stddev);
    cv::Mat src_float; src.convertTo(src_float, CV_32F);
    cv::Mat dst_float = src_float + noise;
    dst_float.convertTo(dst, CV_8U);
}

void addSaltAndPepperNoise(const cv::Mat& src, cv::Mat& dst, double noise_ratio) {
    dst = src.clone();
    int noise_pixels = static_cast<int>(src.rows * src.cols * noise_ratio);
    std::random_device rd; std::mt19937 gen(rd());
    std::uniform_int_distribution<> disX(0, src.cols - 1);
    std::uniform_int_distribution<> disY(0, src.rows - 1);
    for (int i = 0; i < noise_pixels; ++i) {
        dst.at<uchar>(disY(gen), disX(gen)) = (i % 2 == 0) ? 0 : 255;
    }
}

// 3. Frekans Filtresi ve Ters DFT (IDFT) Fonksiyonu
cv::Mat filterAndIDFT(const cv::Mat& noisy_img, double cutoff_radius) {
    cv::Mat padded; 
    noisy_img.convertTo(padded, CV_32F);
    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexI; 
    cv::merge(planes, 2, complexI);
    
    // İleri DFT
    cv::dft(complexI, complexI);
    
    // Merkez Kaydırma (Shift)
    complexI = complexI(cv::Rect(0, 0, complexI.cols & -2, complexI.rows & -2));
    int cx = complexI.cols / 2;
    int cy = complexI.rows / 2;
    cv::Mat q0(complexI, cv::Rect(0, 0, cx, cy)), q1(complexI, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(complexI, cv::Rect(0, cy, cx, cy)), q3(complexI, cv::Rect(cx, cy, cx, cy)), tmp;
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);
    
    // İdeal Alçak Geçiren Filtre Uygulaması (Yüksek frekanslı gürültüleri kes)
    for (int y = 0; y < complexI.rows; y++) {
        for (int x = 0; x < complexI.cols; x++) {
            double distance = std::sqrt(std::pow(x - cx, 2) + std::pow(y - cy, 2));
            if (distance > cutoff_radius) {
                complexI.at<cv::Vec2f>(y, x)[0] = 0.0f; // Reel
                complexI.at<cv::Vec2f>(y, x)[1] = 0.0f; // Sanal
            }
        }
    }
    
    // Merkezi geri kaydır (Inverse Shift)
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);
    
    // Ters DFT (IDFT)
    cv::Mat inverseTransform;
    cv::idft(complexI, inverseTransform, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    
    cv::Mat restored;
    inverseTransform.convertTo(restored, CV_8U);
    return restored;
}

// 4. Zero Mean Normalized Cross Correlation (ZNCC) Hesaplama Fonksiyonu
double calculateZNCC(const cv::Mat& orig, const cv::Mat& recon) {
    cv::Mat mat1, mat2;
    orig.convertTo(mat1, CV_64F);
    recon.convertTo(mat2, CV_64F);
    
    cv::Scalar mean1, stddev1, mean2, stddev2;
    cv::meanStdDev(mat1, mean1, stddev1);
    cv::meanStdDev(mat2, mean2, stddev2);
    
    // Sıfır ortalama (Zero Mean) işlemi
    mat1 -= mean1[0];
    mat2 -= mean2[0];
    
    // Çapraz Korelasyon (Kovaryans) ve Varyans hesapları
    double covar = mat1.dot(mat2);
    double var1 = mat1.dot(mat1);
    double var2 = mat2.dot(mat2);
    
    return covar / std::sqrt(var1 * var2);
}

int main() {
    cv::Mat original = cv::imread("orman_web.jpg", cv::IMREAD_GRAYSCALE);
    if(original.empty()) return -1;
    cv::Mat cropped = cropToPowerOf2(original);

    // Gauss ve Salt & Pepper Gürültüleri Üretimi
    cv::Mat img_gauss, img_sp;
    addGaussianNoise(cropped, img_gauss, 0.0, 40.0);
    addSaltAndPepperNoise(cropped, img_sp, 0.10);

    // DFT üzerinden filtreleyip geri döndürme (Cutoff Radius = 60 piksel)
    double cutoff = 60.0;
    cv::Mat recon_gauss = filterAndIDFT(img_gauss, cutoff);
    cv::Mat recon_sp = filterAndIDFT(img_sp, cutoff);

    // ZNCC Hesaplamaları
    double zncc_gauss_noisy = calculateZNCC(cropped, img_gauss);
    double zncc_gauss_recon = calculateZNCC(cropped, recon_gauss);
    double zncc_sp_noisy = calculateZNCC(cropped, img_sp);
    double zncc_sp_recon = calculateZNCC(cropped, recon_sp);

    // Sonuçları Ekrana Yazdır
    std::cout << "--- ZNCC METRIK SONUCLARI (Maksimum Benzerlik = 1.0) ---" << std::endl;
    std::cout << "Gauss Gurultulu Görüntü ZNCC     : " << zncc_gauss_noisy << std::endl;
    std::cout << "Gauss Filtrelenmis (IDFT) ZNCC   : " << zncc_gauss_recon << std::endl;
    std::cout << "Salt&Pepper Gurultulu ZNCC       : " << zncc_sp_noisy << std::endl;
    std::cout << "Salt&Pepper Filtrelenmis ZNCC    : " << zncc_sp_recon << std::endl;

    // Görüntüleri Kaydet
    cv::Mat c_gauss; cv::hconcat(img_gauss, recon_gauss, c_gauss);
    cv::Mat c_sp; cv::hconcat(img_sp, recon_sp, c_sp);
    cv::imwrite("1_gauss_idft.jpg", c_gauss);
    cv::imwrite("2_sp_idft.jpg", c_sp);

    return 0;
}
