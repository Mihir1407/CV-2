#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <cmath>
#include "opencv2/opencv.hpp"
#include "featureExtraction.h" // Adjust this include path as necessary

// Function to read feature vectors from the CSV file
int readFeatureVectors(const std::string& filename, std::vector<std::string>& imageFilenames, std::vector<std::vector<float>>& featureVectors) {
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string imgFilename;
        std::getline(iss, imgFilename, ',');

        std::vector<float> features;
        float feature;
        while (iss >> feature) {
            features.push_back(feature);
            if (iss.peek() == ',') iss.ignore();
        }

        imageFilenames.push_back(imgFilename);
        featureVectors.push_back(features);
    }

    return 0;
}

// Compute the Euclidean distance between two feature vectors
float computeEuclideanDistance(const std::vector<float>& v1, const std::vector<float>& v2) {
    float sum = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        float diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <target_image_path> <feature_vectors_file> <top_n_matches>\n";
        return -1;
    }

    std::string targetImagePath = argv[1];
    std::string featureVectorsFile = argv[2];
    int topN = std::stoi(argv[3]);

    cv::Mat targetImage = cv::imread(targetImagePath);
    if (targetImage.empty()) {
        std::cerr << "Failed to load target image.\n";
        return -1;
    }

    // Extract combined color and texture features for the target image
    std::vector<float> targetFeatures = extractCombinedFeatures(targetImage);

    std::vector<std::string> imageFilenames;
    std::vector<std::vector<float>> featureVectors;
    readFeatureVectors(featureVectorsFile, imageFilenames, featureVectors);

    std::vector<std::pair<float, std::string>> distances;
    for (size_t i = 0; i < featureVectors.size(); ++i) {
        float distance = computeEuclideanDistance(targetFeatures, featureVectors[i]);
        distances.push_back(std::make_pair(distance, imageFilenames[i]));
    }

    // Sort the distances in ascending order
    std::sort(distances.begin(), distances.end());

    // Find and print the self-match with its score
    auto selfMatchIt = std::find_if(distances.begin(), distances.end(), [&](const std::pair<float, std::string>& result) {
        return std::filesystem::path(result.second).filename() == std::filesystem::path(targetImagePath).filename();
    });
    if (selfMatchIt != distances.end()) {
        std::cout << "Match with itself: " << selfMatchIt->second << " with distance " << selfMatchIt->first << "\n";
        distances.erase(selfMatchIt); // Remove self match from the results
    }

    // Output the top N matches
    for (int i = 0; i < topN && i < distances.size(); ++i) {
        std::cout << "Match " << i + 1 << ": " << distances[i].second << " with distance " << distances[i].first << "\n";
    }

    return 0;
}
