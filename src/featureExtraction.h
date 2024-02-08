// featureExtraction.h
// Name: Mihir Chitre, Aditya Gurnani
// Date: 02/01/2024
// Purpose: Include file for featureExtraction.cpp, includes different functions to extract feature vectors from an image.

#ifndef FEATURE_EXTRACTION_H
#define FEATURE_EXTRACTION_H

#include "opencv2/opencv.hpp"
#include <vector>

std::vector<float> extractFeatureVector(const cv::Mat& image);

// Add this new function declaration
std::vector<float> extractColorHistogram(const cv::Mat& image);

std::vector<float> extractRGBHistograms(const cv::Mat& image);

std::vector<float> extractWholeHistogram(const cv::Mat& image);

std::vector<float> extractTextureFeatures(const cv::Mat& image);

std::vector<float> extractCombinedFeatures(const cv::Mat& image);

#endif 
