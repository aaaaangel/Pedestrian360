#ifndef PANORAMICCAMERA_MYSEAMFINDER_H
#define PANORAMICCAMERA_MYSEAMFINDER_H

#include <iostream>
#include <algorithm>
#include <ctime>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

//#include <torch/script.h>

#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <errno.h>
#include <unistd.h>
#include <stdio.h>



#include "Consts.h"


namespace pc {
    class MySeamFinder {
    public:
        std::vector<cv::Mat> maskMaps_;
        std::vector<cv::Mat> overlapMaskMaps_;
        std::vector<std::vector<std::vector<int> > > overlapRowCols_;
        std::vector<std::vector<int> > lastframe_seamPos_;
        std::vector<std::vector<int> > lastframe_seamItensity_;
        std::vector<cv::Mat> lastframe_seamPosMaps;
        bool not_firstframe=false;

        int (*spacial_cost)[pc::resize_height][pc::resize_width] = new int[4][pc::resize_height][pc::resize_width];
        int (*spacial_mincost_lastcol)[pc::resize_height][pc::resize_width] = new int[4][pc::resize_height][pc::resize_width];
        int (*remapImgs_gray)[pc::resize_height][pc::resize_width] = new int[4][pc::resize_height][pc::resize_width];
        int (*remapImgs_gray_diff_lr)[pc::resize_height][pc::resize_width] = new int[4][pc::resize_height][pc::resize_width];
        int (*remapImgs_gray_diff_ud)[pc::resize_height][pc::resize_width] = new int[4][pc::resize_height][pc::resize_width];
        int (*saliency)[pc::resize_height][pc::resize_width] = new int[4][pc::resize_height][pc::resize_width];

//        torch::jit::script::Module human_module;
        std::vector<cv::Mat> human_saliency;
        static const int total_size = 500*1200*3*1*1;
        unsigned char recvb[total_size/3];

        int cli_sockfd;/*客户端SOCKET */
        void initialize_socket();


        MySeamFinder(std::vector<cv::UMat> &maskMaps);

        bool
        find_dp_temporal_fast(std::vector<cv::UMat> &remapImgs, std::vector<cv::UMat> &maskMapsSeam, bool human=false);

        void translateTransform_x(cv::Mat const &src, cv::Mat &dst, int dx);

        void translateTransform_y(cv::Mat const &src, cv::Mat &dst, int dy);

        void human_segmentation_socket(std::vector<cv::UMat> &remapImgs);
    };
}

#endif //PANORAMICCAMERA_MYSEAMFINDER_H
