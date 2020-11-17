#ifndef PANORAMICCAMERA_PROJECTOR_H
#define PANORAMICCAMERA_PROJECTOR_H

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv_modules.hpp>

#include "CameraParameter.h"
#include "Consts.h"

namespace pc{
    class Projector {
    public:
        CameraParameter camera_;

        Projector(CameraParameter camera):camera_(camera){};
        virtual bool mapBackward(double a, double b, double &x, double &y)=0;
    };


    class CylinderProjector : public Projector{
    public:
        CylinderProjector(CameraParameter camera, double radius)
        :Projector(camera), radius_(radius)
        {
            u_step_ = 2 * PI / (pc::stitchResultSize.width);
            height_scale_ = 2 * PI * radius_ / pc::stitchResultSize.width;
        }

        double radius_;
        double height_scale_;
        double u_step_;

        bool mapBackward(double a, double b, double &x, double &y);
    };

    class SphereProjector : public Projector{
    public:
        SphereProjector(CameraParameter camera, double radius)
                :Projector(camera), radius_(radius)
        {
            u_step_ = 2 * PI / (pc::stitchResultSize.width);
            v_step_ = PI / (pc::stitchResultSize.height);
        }

        double radius_;
        double u_step_;
        double v_step_;

        bool mapBackward(double a, double b, double &x, double &y);
    };

    class BucketProjector : public Projector{
    public:
        BucketProjector(CameraParameter camera, double radius)
                :Projector(camera), radius_(radius)
        {
            u_step_ = 2 * PI / (pc::stitchResultSize.width);
            height_scale_ = 2 * PI * radius_ / pc::stitchResultSize.width;
        }

        double radius_;
        double u_step_;
        double height_scale_;
        double bottem_ = 0 * (pc::stitchResultSize.height/2);  //[-v/2, v/2] v=pc::stitchResultSize.height

        bool mapBackward(double a, double b, double &x, double &y);
    };

}


#endif //PANORAMICCAMERA_PROJECTOR_H
