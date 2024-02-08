#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <cmath>
#include "opencv2/opencv.hpp"
#include "featureExtraction.h"


// Function to calculate the L2 norm of a vector
float l2Norm(const std::vector<float>& vec) {
    float sum = 0.0f;
    for (const auto& val : vec) {
        sum += val * val;
    }
    return std::sqrt(sum);
}

// Function to calculate cosine distance between two vectors
float cosineDistance(const std::vector<float>& v1, const std::vector<float>& v2) {
    float dotProduct = 0.0f, normV1 = 0.0f, normV2 = 0.0f;
    for (size_t i = 0; i < v1.size(); ++i) {
        dotProduct += v1[i] * v2[i];
    }
    normV1 = l2Norm(v1);
    normV2 = l2Norm(v2);

    // Avoid division by zero
    if (normV1 == 0.0f || normV2 == 0.0f) {
        return 1.0f; // Max distance in case of zero vector
    }
    
    return 1.0f - (dotProduct / (normV1 * normV2));
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

// Include the functions to read CSV and calculate cosine distance here

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <target_image_filename> <feature_vectors_file> <top_n_matches>\n";
        return -1;
    }

    std::string targetImageFilename = argv[1];
    std::string featureVectorsFile = argv[2];
    int topN = std::stoi(argv[3]);

    // Read the feature vectors and filenames from the CSV
    std::vector<std::string> imageFilenames;
    std::vector<std::vector<float>> featureVectors;
    readFeatureVectors(featureVectorsFile, imageFilenames, featureVectors);

    // Find the feature vector for the target image
    auto it = std::find(imageFilenames.begin(), imageFilenames.end(), targetImageFilename);
    if (it == imageFilenames.end()) {
        std::cerr << "Target image features not found in the file.\n";
        return -1;
    }
    int index = std::distance(imageFilenames.begin(), it);
    std::vector<float> targetFeatures = featureVectors[index];

    // Calculate distances and sort them
    std::vector<std::pair<float, std::string>> distances;
    for (size_t i = 0; i < featureVectors.size(); ++i) {
        if (i != index) { // Skip the target image itself
            float distance = cosineDistance(targetFeatures, featureVectors[i]);
            distances.push_back(std::make_pair(distance, imageFilenames[i]));
        }
    }
    
    std::sort(distances.begin(), distances.end());

    // Output the top N matches
    for (int i = 0; i < topN && i < distances.size(); ++i) {
        std::cout << "Match " << i + 1 << ": " << distances[i].second << " with distance " << distances[i].first << "\n";
    }

    return 0;
}
