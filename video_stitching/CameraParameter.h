#ifndef PANORAMICCAMERA_CAMERAPARAMETER_H
#define PANORAMICCAMERA_CAMERAPARAMETER_H

#include<iostream>
#include <opencv2/core/core.hpp>

namespace pc{
    class CameraParameter {
    public:
        CameraParameter(){}
        CameraParameter(std::string ParameterFileName);
        cv::Matx33d K;
        cv::Vec4d D;
        cv::Matx33d R;
        cv::Vec3d t;
    };
}




#endif //PANORAMICCAMERA_CAMERAPARAMETER_H
