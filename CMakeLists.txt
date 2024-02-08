cmake_minimum_required(VERSION 3.10)

project(FirstProject2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(OpenCV_DIR C:/opencv/build)

find_package(OpenCV REQUIRED)

# Added executable for imgDisplay.cpp
add_executable(readImages src/readImages.cpp src/featureExtraction.cpp)
target_link_libraries(readImages ${OpenCV_LIBS})

# Added executable for imgDisplay.cpp
add_executable(matchImagesBaseline src/matchImagesBaseline.cpp src/featureExtraction.cpp)
target_link_libraries(matchImagesBaseline ${OpenCV_LIBS})

# Added executable for imgDisplay.cpp
add_executable(matchImagesHistogram src/matchImagesHistogram.cpp src/featureExtraction.cpp)
target_link_libraries(matchImagesHistogram ${OpenCV_LIBS})

# Added executable for imgDisplay.cpp
add_executable(matchImagesMultiHistogram src/matchImagesMultiHistogram.cpp src/featureExtraction.cpp)
target_link_libraries(matchImagesMultiHistogram ${OpenCV_LIBS})

# Added executable for imgDisplay.cpp
add_executable(matchImagesColorTexture src/matchImagesColorTexture.cpp src/featureExtraction.cpp)
target_link_libraries(matchImagesColorTexture ${OpenCV_LIBS})

# Added executable for imgDisplay.cpp
add_executable(matchImagesDeepNetwork src/matchImagesDeepNetwork.cpp src/featureExtraction.cpp)
target_link_libraries(matchImagesDeepNetwork ${OpenCV_LIBS})

# Added executable for imgDisplay.cpp
add_executable(colors src/colors.cpp src/kmeans.cpp)
target_link_libraries(colors ${OpenCV_LIBS})

# # Added executable for VidDisplay.cpp
# add_executable(VidDisplay src/vidDisplay.cpp src/filter.cpp src/faceDetect.cpp)
# target_link_libraries(VidDisplay ${OpenCV_LIBS})
