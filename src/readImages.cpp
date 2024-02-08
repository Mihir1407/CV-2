// readImages.cpp
// Name: Mihir Chitre, Aditya Gurnani
// Date: 02/01/2024
// Purpose: Reads all the images in the given directory and generates an output csv file containing feature vectors for each image, using
//          the selected feature set.

#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <filesystem>
#include "opencv2/opencv.hpp"
#include "featureExtraction.h"

// Function to append image data to a CSV file
int append_image_data_csv(const char *filename, const char *image_filename, std::vector<float> &image_data, int reset_file = 0)
{
    char mode[2] = {reset_file ? 'w' : 'a', '\0'};
    FILE *fp = fopen(filename, mode);
    if (!fp)
    {
        printf("Unable to open output file %s\n", filename);
        return -1;
    }

    std::filesystem::path p(image_filename);
    std::string filenameOnly = p.filename().string();

    fprintf(fp, "%s", filenameOnly.c_str());
    for (float value : image_data)
    {
        fprintf(fp, ",%.4f", value);
    }
    fprintf(fp, "\n");
    fclose(fp);

    return 0;
}

// Main function to process images in a directory and write feature vectors to CSV
int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        printf("Usage: %s <directory> <output_csv_file> <feature_extraction_method>\n", argv[0]);
        return -1;
    }

    const std::string directory = argv[1];
    const char *output_csv = argv[2];
    std::string featureExtractionMethod = argv[3];

    std::vector<float> (*featureExtractionFunction)(const cv::Mat &);
    if (featureExtractionMethod == "baseline")
    {
        featureExtractionFunction = &extractFeatureVector;
    }
    else if (featureExtractionMethod == "histogramMatching")
    {
        featureExtractionFunction = &extractColorHistogram;
    }
    else if (featureExtractionMethod == "multiHistogramMatching")
    {
        featureExtractionFunction = &extractRGBHistograms;
    }
    else if (featureExtractionMethod == "combinedFeatures")
    {
        featureExtractionFunction = &extractCombinedFeatures;
    }

    bool first_file = true;

    for (const auto &entry : std::filesystem::directory_iterator(directory))
    {
        std::string imagePath = entry.path().string();
        cv::Mat image = cv::imread(imagePath);
        if (image.empty())
        {
            printf("Failed to open image %s\n", imagePath.c_str());
            continue;
        }

        std::vector<float> featureVector = featureExtractionFunction(image);
        append_image_data_csv(output_csv, imagePath.c_str(), featureVector, first_file);
        first_file = false;
    }

    return 0;
}
