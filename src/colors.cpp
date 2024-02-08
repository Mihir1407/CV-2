#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <opencv2/opencv.hpp>
#include "kmeans.h"

cv::Vec3b closestColor(cv::Vec3b &pix, std::vector<cv::Vec3b> &colors) {
  int mincolor = 0;
  int mindist = SSD(pix, colors[0]);

  for(int i = 1; i < colors.size(); i++) {
    int sse = SSD(pix, colors[i]);
    if(sse < mindist) {
      mindist = sse;
      mincolor = i;
    }
  }

  return(colors[mincolor]);
}

int main(int argc, char *argv[]) {
  cv::Mat src;
  cv::Mat dst; // viewing image
  cv::Mat resizedSrc; // resized source image for display
  cv::Mat resizedDst; // resized destination image for display
  char filename[256];
  int ncolors = 16;
  double scaleFactor = 0.5; // Scale factor for resizing the image

  if(argc < 3) {
    printf("usage: %s <image filename> <# of colors>\n", argv[0]);
    return(-1);
  }

  strcpy(filename, argv[1]);
  src = cv::imread(filename);
  if(src.data == NULL) {
    printf("error: unable to read filename %s\n", filename);
    return(-2);
  }

  // Resize the original image for display
  // cv::resize(src, resizedSrc, cv::Size(), scaleFactor, scaleFactor);
  cv::imshow("Original", resizedSrc);

  int tcolors = atoi(argv[2]);
  if(tcolors < 1 || tcolors > 66000) {
    printf("error: number of colors must be in [1, 66000]\n");
  } else {
    ncolors = tcolors;
  }

  std::vector<cv::Vec3b> data;
  int B = 4;
  for(int i = 0; i < src.rows - B; i += B) {
    for(int j = 0; j < src.cols - B; j += B) {
      int jx = rand() % B;
      int jy = rand() % B;
      data.push_back(src.at<cv::Vec3b>(i + jy, j + jx));
    }
  }

  std::vector<cv::Vec3b> means;
  int *labels = new int[data.size()];

  if(kmeans(data, means, labels, ncolors)) {
    printf("Error using kmeans\n");
    return(-1);
  }

  dst.create(src.size(), src.type());
  for(int i = 0; i < src.rows; i++) {
    for(int j = 0; j < src.cols; j++) {
      dst.at<cv::Vec3b>(i, j) = closestColor(src.at<cv::Vec3b>(i, j), means);
    }
  }

  // Resize the clustered image for display
  // cv::resize(dst, resizedDst, cv::Size(), scaleFactor, scaleFactor);
  cv::imshow("clustered", resizedDst);

  cv::waitKey(0);

  delete[] labels;

  return(0);
}
