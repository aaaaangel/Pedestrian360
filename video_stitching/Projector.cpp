#include <opencv2/imgproc.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv_modules.hpp>
#include "Projector.h"

using namespace pc;

bool CylinderProjector::mapBackward(double u, double v, double &x, double &y){
    cv::Point3d global_point;
    double sita = u * u_step_;

    global_point.x = cos(sita) * radius_;
    global_point.y = sin(sita) * radius_;
    global_point.z = (pc::stitchResultSize.height/2-v) * height_scale_;

    auto pointInCameraCoord = camera_.R * cv::Vec3d(global_point) + camera_.t;
    if (pointInCameraCoord(2) <= 0)
        return false;

    //原图像点坐标
    std::vector<cv::Point2f> img_point;
    auto Affine = cv::Affine3d(camera_.R, camera_.t);
    cv::fisheye::projectPoints(std::vector<cv::Point3f>(1, cv::Point3d(global_point.x, global_point.y, global_point.z)), img_point, Affine, camera_.K, camera_.D);
    x = img_point.front().x;
    y = img_point.front().y;
    return true;
}


bool SphereProjector::mapBackward(double u, double v, double &x, double &y){
    cv::Point3d global_point;
    double sita = u * u_step_;
    double alpha = v * v_step_;

    global_point.x = cos(sita) * radius_ * sin(alpha);
    global_point.y = sin(sita) * radius_ * sin(alpha);
    global_point.z = radius_ * cos(alpha);

    auto pointInCameraCoord = camera_.R * cv::Vec3d(global_point) + camera_.t;
    if (pointInCameraCoord(2) <= 0)
        return false;

    //原图像点坐标
    std::vector<cv::Point2f> img_point;
    auto Affine = cv::Affine3d(camera_.R, camera_.t);
    cv::fisheye::projectPoints(std::vector<cv::Point3f>(1, cv::Point3d(global_point.x, global_point.y, global_point.z)), img_point, Affine, camera_.K, camera_.D);
    x = img_point.front().x;
    y = img_point.front().y;
    return true;
}

bool BucketProjector::mapBackward(double u, double v, double &x, double &y){
    cv::Point3d global_point;
    double sita = u * u_step_;

    global_point.x = cos(sita) * radius_;
    global_point.y = sin(sita) * radius_;
    global_point.z = (pc::stitchResultSize.height/2-v);

    if(global_point.z<bottem_){
        double temp = (global_point.z - (- pc::stitchResultSize.height/2 ))/ (bottem_ - (-pc::stitchResultSize.height/2 ));
        temp = temp*temp*temp;
        temp = temp*temp;
        global_point.x *= temp;
        global_point.y *= temp;
        global_point.z = bottem_;
    }
    else{
        global_point.z *= height_scale_;
    }

    auto pointInCameraCoord = camera_.R * cv::Vec3d(global_point) + camera_.t;
    if (pointInCameraCoord(2) <= 0)
        return false;

    //原图像点坐标
    std::vector<cv::Point2f> img_point;
    auto Affine = cv::Affine3d(camera_.R, camera_.t);
    cv::fisheye::projectPoints(std::vector<cv::Point3f>(1, cv::Point3d(global_point.x, global_point.y, global_point.z)), img_point, Affine, camera_.K, camera_.D);
    x = img_point.front().x;
    y = img_point.front().y;
    return true;
}