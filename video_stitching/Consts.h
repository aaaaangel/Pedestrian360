#pragma once
#ifndef PANORAMICCAMERA_CONSTS_H
#define PANORAMICCAMERA_CONSTS_H

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>


namespace pc{
    const int numCamera = 4;
    static char *parm_files[4]={
            "../resources/calibration_data//FAffineLX.txt",
            "../resources/calibration_data//RAffineLX.txt",
            "../resources/calibration_data//BAffineLX.txt",
            "../resources/calibration_data//LAffineLX.txt"};
    const double PI = 3.1415926;
    const cv::Size photoSize(1920, 1080); // 1280 1080
    const cv::Size stitchResultSize(1200, 500);
    const float blendWidth = sqrt(static_cast<float>(stitchResultSize.area())) * 5.0 / 100.0;
    const int numBands = (static_cast<int>(ceil(log(blendWidth) / log(2.0)) - 1.0));
    const double disFloorToCenter = 0.005;


    // seam-cutting in opencv
//    const int seamTypeFirstFrame = cv::detail::SeamFinder::VORONOI_SEAM;
    const int seamTypeFirstFrame = cv::detail::SeamFinder::DP_SEAM;
//    const int seamTypeFirstFrame = cv::detail::SeamFinder::NO;
//    const int seamTypeFirstFrame = 3;

    // speed
    const float reScale =1;
    const cv::Size resizeStitchResultSize(1200*reScale, 500*reScale);
    const int resize_width = 1200*reScale;
    const int resize_height = 500*reScale;
    const int threads_num = 1;

}

#endif //PANORAMICCAMERA_CONSTS_H
