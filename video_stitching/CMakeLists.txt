cmake_minimum_required(VERSION 3.13)
project(PanoramicCamera)

set(CMAKE_CXX_STANDARD 11)

set(OpenCV_DIR "/usr/local/lib/cmake/opencv4/OpenCVConfig.cmake")
set(OpenCV_INCLUDE_DIRS "/usr/local/include/opencv4")
set(OpenCV_LIBRARIES "/usr/local/lib/")
#find_package(OpenCV REQUIRED)
set(OpenCV_INCLUDE_DIRS "/usr/local/include/opencv4")
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories("/usr/local/include/eigen3/")

find_package(Threads)

link_directories("/home/angel/Documents/opencv-4.2.0/build_no_TIFF/lib")

add_executable(MTPanoramicCamera multiThread.cpp CameraParameter.cpp Projector.cpp Sticher.cpp  MySeamFinder.cpp)
target_link_libraries(MTPanoramicCamera  "${TORCH_LIBRARIES}" ${CMAKE_THREAD_LIBS_INIT} opencv_calib3d opencv_core opencv_dnn opencv_features2d opencv_flann opencv_highgui opencv_imgcodecs opencv_imgproc opencv_ml opencv_objdetect opencv_photo opencv_stitching opencv_video opencv_videoio)
