// matchImages.cpp
// Name: Mihir Chitre, Aditya Gurnani
// Date: 02/01/2024
// Purpose: Matches the images in the given directory to the target image, using the file containing the feature vectors and the selected 
//          feature set.

#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <filesystem>
#include "opencv2/opencv.hpp"
#include "featureExtraction.h"

float computeSSD(const std::vector<float>& v1, const std::vector<float>& v2) {
    float sum = 0.0f;
    for (size_t i = 0; i < v1.size(); ++i) {
        float diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    return sum;
}

float histogramIntersection(const std::vector<float>& h1, const std::vector<float>& h2) {
    float intersection = 0.0f;
    for (size_t i = 0; i < h1.size(); ++i) {
        intersection += std::min(h1[i], h2[i]);
    }
    return intersection;
}

int readFeatureVectors(const std::string& filename, std::vector<std::string>& imageFilenames, std::vector<std::vector<float>>& featureVectors) {
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string filename;
        std::getline(iss, filename, ',');

        std::vector<float> features;
        float feature;
        while (iss >> feature) {
            features.push_back(feature);
            if (iss.peek() == ',') {
                iss.ignore();
            }
        }

        imageFilenames.push_back(filename);
        featureVectors.push_back(features);
    }

    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <target_image> <feature_csv_file> <top_n_matches>\n";
        return -1;
    }

    std::string targetImagePath = argv[1];
    std::string csvFile = argv[2];
    int topN = std::stoi(argv[3]);

    std::vector<float> (*featureExtractionFunction)(const cv::Mat&);
    featureExtractionFunction = &extractFeatureVector;

    cv::Mat targetImage = cv::imread(targetImagePath);
    if (targetImage.empty()) {
        std::cerr << "Failed to open target image " << targetImagePath << "\n";
        return -1;
    }

    std::filesystem::path targetPath(targetImagePath);
    std::string targetFilename = targetPath.filename().string();

    std::vector<float> targetFeature = featureExtractionFunction(targetImage);

    std::vector<std::string> imageFilenames;
    std::vector<std::vector<float>> featureVectors;
    readFeatureVectors(csvFile, imageFilenames, featureVectors);

    std::vector<std::pair<float, std::string>> ssdResults;
    for (size_t i = 0; i < featureVectors.size(); ++i) {
        float ssd = computeSSD(targetFeature, featureVectors[i]);
        ssdResults.emplace_back(ssd, imageFilenames[i]);
    }

    // Sort the results based on SSD
    std::sort(ssdResults.begin(), ssdResults.end());

    // Print match with itself (distance 0) without counting it in top N
    auto it = std::find_if(ssdResults.begin(), ssdResults.end(), [&targetFilename](const std::pair<float, std::string>& element) {
        return element.second == targetFilename;
    });

    if (it != ssdResults.end()) {
        std::cout << "Match with itself: " << it->second << " with distance " << it->first << "\n";
        // Remove the match with itself so it doesn't count in the top N
        ssdResults.erase(it);
    }

    // Output the top N matches excluding the match with itself
    for (int i = 0; i < topN && i < ssdResults.size(); ++i) {
        std::cout << "Match " << i + 1 << ": " << ssdResults[i].second << " with distance " << ssdResults[i].first << "\n";
    }

    return 0;
}
