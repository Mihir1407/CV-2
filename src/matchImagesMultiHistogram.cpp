#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include "opencv2/opencv.hpp"
#include "featureExtraction.h" // Ensure it includes extractMultiPartHistogram

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

float histogramIntersectionDistance(const std::vector<float>& h1, const std::vector<float>& h2) {
    float intersection = 0.0f;
    for (size_t i = 0; i < h1.size(); i++) {
        intersection += std::min(h1[i], h2[i]);
    }
    return intersection;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <target_image_path> <feature_vectors_file> <top_n_matches>\n";
        return -1;
    }

    std::string targetImagePath = argv[1], featureVectorsFile = argv[2];
    int topN = std::stoi(argv[3]);

    cv::Mat targetImage = cv::imread(targetImagePath);
    if (targetImage.empty()) {
        std::cerr << "Failed to load target image.\n";
        return -1;
    }

    std::vector<float> targetFeatures = extractRGBHistograms(targetImage);

    // Assuming you have a function to read feature vectors from other images
    // and it populates a vector of filenames and a vector of corresponding feature vectors
    std::vector<std::string> imageFilenames;
    std::vector<std::vector<float>> featureVectors;
    // You should implement the readFeatureVectors function based on your dataset
    readFeatureVectors(featureVectorsFile, imageFilenames, featureVectors);

    // Compute histogram intersection distances
    std::vector<std::pair<float, std::string>> scores;
    for (size_t i = 0; i < featureVectors.size(); ++i) {
        float score = histogramIntersectionDistance(targetFeatures, featureVectors[i]);
        scores.push_back({score, imageFilenames[i]});
    }

    // Sort by descending score as we use intersection (higher is more similar)
    std::sort(scores.begin(), scores.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });

    // Find and print the self-match with its score
    auto selfMatchIt = std::find_if(scores.begin(), scores.end(), [&](const std::pair<float, std::string>& result) {
        return std::filesystem::path(result.second).filename() == std::filesystem::path(targetImagePath).filename();
    });
    if (selfMatchIt != scores.end()) {
        std::cout << "Match with itself: " << selfMatchIt->second << " with score " << selfMatchIt->first << "\n";
        scores.erase(selfMatchIt); // Remove self match from the results
    }

    // Output the top N matches, now excluding the self match
    for (int i = 0; i < topN && i < scores.size(); ++i) {
        std::cout << "Match " << i + 1 << ": " << scores[i].second << " with score " << scores[i].first << "\n";
    }

    return 0;
}