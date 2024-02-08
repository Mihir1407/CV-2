// featureExtraction.cpp
// Name: Mihir Chitre, Aditya Gurnani
// Date: 02/01/2024
// Purpose: Includes different functions to extract feature vectors from an image.

#include "featureExtraction.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

void customCalcHist(const cv::Mat& image, std::vector<float>& hist, int bins, float rangeStart, float rangeEnd) {
    hist.assign(bins, 0.0f);
    float binWidth = (rangeEnd - rangeStart) / bins;
    int pixels = 0;

    // Check the depth of the image to determine how to access pixel values
    if (image.depth() == CV_8U) {
        for (int y = 0; y < image.rows; y++) {
            for (int x = 0; x < image.cols; x++) {
                uchar val = image.at<uchar>(y, x); // Accessing pixel as uchar for 8-bit image
                int bin = static_cast<int>((val - rangeStart) / binWidth);
                if (bin >= 0 && bin < bins) {
                    hist[bin] += 1.0f;
                }
            }
        }
    } else if (image.depth() == CV_32F) {
        for (int y = 0; y < image.rows; y++) {
            for (int x = 0; x < image.cols; x++) {
                float val = image.at<float>(y, x); // Accessing pixel as float for floating-point image
                int bin = static_cast<int>((val - rangeStart) / binWidth);
                if (bin >= 0 && bin < bins) {
                    hist[bin] += 1.0f;
                }
            }
        }
    }

    // Normalize the histogram to sum up to 1
    for (auto &val : hist) {
        val /= image.total();
    }
}

void customNormalizeMinMax(std::vector<float>& hist, float lower, float upper) {
    float sum = 0.0f;
    for (auto val : hist) {
        sum += val;
    }

    if (sum > 0) {
        for (auto &val : hist) {
            val = (val / sum) * (upper - lower) + lower;
        }
    }
}

void customNormalizeL1(std::vector<float>& hist, float lower, float upper) {
    float sum = 0.0f;
    for (auto val : hist) {
        sum += val;
    }

    if (sum > 0) {
        for (auto &val : hist) {
            val = (val / sum) * (upper - lower) + lower;
        }
    }
}

// Function to extract a 7x7 feature vector from the center of each channel of the image
std::vector<float> extractFeatureVector(const cv::Mat& image) {
    int centerX = image.cols / 2;
    int centerY = image.rows / 2;
    int size = 7;

    cv::Rect roi(centerX - size/2, centerY - size/2, size, size);

    std::vector<cv::Mat> channels;
    cv::split(image, channels);

    std::vector<float> featureVector;
    featureVector.reserve(size * size * 3); 

    for (const auto& channel : channels) {
        cv::Mat cropped = channel(roi).clone();
        for (int y = 0; y < cropped.rows; y++) {
            for (int x = 0; x < cropped.cols; x++) {
                featureVector.push_back(static_cast<float>(cropped.at<uchar>(y, x)));
            }
        }
    }

    return featureVector;
}

std::vector<float> extractColorHistogram(const cv::Mat& image) {
    // Convert image to float and normalize to 1
    cv::Mat imageFloat;
    image.convertTo(imageFloat, CV_32F, 1.0/255);

    // Calculate r and g chromaticity
    std::vector<cv::Mat> channels(3);
    cv::split(imageFloat, channels);
    cv::Mat r = channels[2] / (channels[0] + channels[1] + channels[2] + 1e-6); // Avoid division by zero
    cv::Mat g = channels[1] / (channels[0] + channels[1] + channels[2] + 1e-6);

    // Compute the histogram for r and g channels
    int bins = 16; // 16 bins for each dimension
    std::vector<float> histR, histG;
    customCalcHist(r, histR, bins, 0, 1); // Compute histogram for r chromaticity
    customCalcHist(g, histG, bins, 0, 1); // Compute histogram for g chromaticity

    // Normalize the histograms
    customNormalizeMinMax(histR, 0, 1);
    customNormalizeMinMax(histG, 0, 1);

    // Combine r and g histograms into a single vector
    std::vector<float> histVector;
    histVector.insert(histVector.end(), histR.begin(), histR.end());
    histVector.insert(histVector.end(), histG.begin(), histG.end());

    return histVector;
}

std::vector<float> extractRGBHistograms(const cv::Mat& image) {
    std::vector<float> featureVector;
    int bins = 8; // Number of bins for histogram

    // Define halves
    cv::Rect topHalf(0, 0, image.cols, image.rows / 2);
    cv::Rect bottomHalf(0, image.rows / 2, image.cols, image.rows / 2);

    // Ensure we're working with an 8-bit image
    cv::Mat image8u;
    if (image.depth() != CV_8U) {
        image.convertTo(image8u, CV_8U); // Convert to 8-bit if not already
    } else {
        image8u = image;
    }

    // Process each half
    for (const auto& region : {topHalf, bottomHalf}) {
        cv::Mat roi = image8u(region);
        std::vector<cv::Mat> channels;
        cv::split(roi, channels); // Split into color channels

        for (int i = 0; i < 3; i++) { // For each color channel
            std::vector<float> hist;
            customCalcHist(channels[i], hist, bins, 0, 256); // Calculate histogram
            customNormalizeL1(hist, 0, 1); // Normalize the histogram

            // Append histogram data to feature vector
            featureVector.insert(featureVector.end(), hist.begin(), hist.end());
        }
    }

    return featureVector;
}

std::vector<float> extractWholeHistogram(const cv::Mat& image) {
    int binsPerChannel = 150; // 100 bins for each RGB channel
    std::vector<float> featureVector;

    // Assuming image is already in RGB format. If not, convert it using cv::cvtColor if needed.
    std::vector<cv::Mat> channels(3);
    cv::split(image, channels); // Split the image into its color channels

    // Compute and normalize histogram for each channel
    for (int i = 0; i < 3; i++) { // Iterate over RGB channels
        std::vector<float> hist;
        customCalcHist(channels[i], hist, binsPerChannel, 0, 256);
        customNormalizeL1(hist, 0, 1); // Normalize using L1 norm
        featureVector.insert(featureVector.end(), hist.begin(), hist.end());
    }

    return featureVector;
}


std::vector<float> extractTextureFeatures(const cv::Mat& image) {
    cv::Mat gray, grad_x, grad_y;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);

    cv::Mat magnitude, angle;
    cv::cartToPolar(grad_x, grad_y, magnitude, angle, true);

    int magBins = 112, angleBins = 113;
    std::vector<float> magHist, angleHist;
    std::vector<float> featureVector;

    // Compute histogram for magnitude
    customCalcHist(magnitude, magHist, magBins, 0, 256);
    customNormalizeL1(magHist, 0, 1); // Normalize using L1 norm

    // Compute histogram for angle
    customCalcHist(angle, angleHist, angleBins, 0, 360);
    customNormalizeL1(angleHist, 0, 1); // Normalize using L1 norm

    // Combine histograms
    featureVector.insert(featureVector.end(), magHist.begin(), magHist.end());
    featureVector.insert(featureVector.end(), angleHist.begin(), angleHist.end());

    return featureVector;
}

// Combine Color and Texture Features
std::vector<float> extractCombinedFeatures(const cv::Mat& image) {
    std::vector<float> colorFeatures = extractWholeHistogram(image);
    std::vector<float> textureFeatures = extractTextureFeatures(image);

    colorFeatures.insert(colorFeatures.end(), textureFeatures.begin(), textureFeatures.end());
    return colorFeatures;
}