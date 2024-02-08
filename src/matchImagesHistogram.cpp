// matchImagesHistogram.cpp
// Purpose: Matches images using histogram intersection for histogram-based features.

#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <filesystem>
#include "opencv2/opencv.hpp"
#include "featureExtraction.h" 

// Function to compute histogram intersection, higher values indicate more similarity
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

    // Setup similar to matchImages.cpp but tailored for histogram features
    std::string targetImagePath = argv[1];
    std::string csvFile = argv[2];
    int topN = std::stoi(argv[3]);

    // Assuming the method is for histogram matching, we directly use extractColorHistogram
    cv::Mat targetImage = cv::imread(targetImagePath);
    if (targetImage.empty()) {
        std::cerr << "Failed to open target image " << targetImagePath << "\n";
        return -1;
    }

    std::vector<float> targetFeature = extractColorHistogram(targetImage);

    std::vector<std::string> imageFilenames;
    std::vector<std::vector<float>> featureVectors;
    readFeatureVectors(csvFile, imageFilenames, featureVectors);

    // Calculate intersections and store results
    std::vector<std::pair<float, std::string>> intersectionResults;
    for (size_t i = 0; i < featureVectors.size(); ++i) {
        float intersection = histogramIntersection(targetFeature, featureVectors[i]);
        intersectionResults.emplace_back(intersection, imageFilenames[i]);
    }

    // Sort results based on intersection, descending order to prioritize higher values
    std::sort(intersectionResults.begin(), intersectionResults.end(), [](const auto& a, const auto& b) {
        return a.first > b.first; // Note the change here for descending order
    });

// Find and print the self-match with its intersection score
    auto selfMatchIt = std::find_if(intersectionResults.begin(), intersectionResults.end(), [&](const std::pair<float, std::string>& result) {
        return std::filesystem::path(result.second).filename() == std::filesystem::path(targetImagePath).filename();
    });
    if (selfMatchIt != intersectionResults.end()) {
        std::cout << "Match with itself: " << selfMatchIt->second << " with intersection " << selfMatchIt->first << "\n";
        intersectionResults.erase(selfMatchIt);
    }

    // Output the top N matches, including the self match
    for (int i = 0; i < topN && i < intersectionResults.size(); ++i) {
        // No need to skip the self match this time, as it's already been addressed
        std::cout << "Match " << i + 1 << ": " << intersectionResults[i].second << " with intersection " << intersectionResults[i].first << "\n";
    }

    return 0;
}
