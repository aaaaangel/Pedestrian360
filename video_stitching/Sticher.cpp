#include "Sticher.h"

#include <string>


pc::Sticher::Sticher(){
    // init
    std::vector<pc::CameraParameter> cameras;
    for(int i=0;i<pc::numCamera;i++){
        pc::CameraParameter camera = (std::string(pc::parm_files[i]));
        cameras.push_back(camera);
        projectors.push_back(pc::CylinderProjector(camera, 24/pc::disFloorToCenter));
        xMaps.push_back(cv::Mat(pc::stitchResultSize, CV_32F));
        yMaps.push_back(cv::Mat(pc::stitchResultSize, CV_32F));
        maskMaps.push_back(cv::UMat::zeros(pc::stitchResultSize, CV_8U));
        maskMapsSeam.push_back(cv::UMat::zeros(pc::stitchResultSize, CV_8U));
        resizeMaskMapsSeam.push_back(cv::UMat::zeros(pc::resizeStitchResultSize, CV_8U));
        resizeMaskMaps.push_back(cv::UMat(pc::resizeStitchResultSize, CV_8U));
        corners.push_back(cv::Point(0,0));
    }

    // compute xmap ymap and mask
    double x,y;
    for(int k=0;k<pc::numCamera;k++){
        cv::Mat maskMaps_temp = maskMaps[k].getMat(cv::ACCESS_WRITE);
        for(int i=0; i<pc::stitchResultSize.width; i++){
            for(int j=0; j<pc::stitchResultSize.height; j++){
                if(projectors[k].mapBackward(i,j,x,y) &&
                   x>=double(0) && x<pc::photoSize.width && y>=double(0) && y<pc::photoSize.height){
                    xMaps[k].at<float>(j,i) = x;
                    yMaps[k].at<float>(j,i) = y;
                    maskMaps_temp.at<uchar>(j,i) = 255;
                }
            }
        }
    }

    cv::bitwise_and(maskMaps[0], maskMaps[1], overlapFR, maskMaps[0]);
    cv::bitwise_and(maskMaps[1], maskMaps[2], overlapRB, maskMaps[1]);
    cv::bitwise_and(maskMaps[2], maskMaps[3], overlapBL, maskMaps[2]);
    cv::bitwise_and(maskMaps[3], maskMaps[0], overlapLF, maskMaps[3]);

    // init seam
    for(int i=0;i<pc::numCamera;i++) {
        cv::resize(maskMaps[i], resizeMaskMaps[i], pc::resizeStitchResultSize);
    }

    mySeamFinder = new pc::MySeamFinder(resizeMaskMaps);

    for(int i=0; i<3; i++)
        photometricAlignment_parameters.push_back(Eigen::Vector4d(1.0 , 1.0 , 1.0 , 1.0));
}


void pc::Sticher::photometricAlignment_std(cv::UMat &imgF, cv::UMat &imgR, cv::UMat &imgB, cv::UMat &imgL) {

    cv::Mat F_YCbCr, L_YCbCr, B_YCbCr, R_YCbCr;
    cv::cvtColor(imgF, F_YCbCr, cv::COLOR_BGR2YCrCb);
    cv::cvtColor(imgR, R_YCbCr, cv::COLOR_BGR2YCrCb);
    cv::cvtColor(imgB, B_YCbCr, cv::COLOR_BGR2YCrCb);
    cv::cvtColor(imgL, L_YCbCr, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> F_channels, L_channels, B_channels, R_channels;
    cv::split(F_YCbCr, F_channels);
    cv::split(R_YCbCr, R_channels);
    cv::split(B_YCbCr, B_channels);
    cv::split(L_YCbCr, L_channels);

    cv::Scalar meanFR_F, meanFR_R, meanRB_R, meanRB_B, meanBL_B, meanBL_L, meanLF_L, meanLF_F;
    cv::Scalar stdFR_F, stdFR_R, stdRB_R, stdRB_B, stdBL_B, stdBL_L, stdLF_L, stdLF_F;
    cv::meanStdDev(F_YCbCr, meanFR_F, stdFR_F, overlapFR);
    cv::meanStdDev(R_YCbCr, meanFR_R, stdFR_R, overlapFR);
    cv::meanStdDev(R_YCbCr, meanRB_R, stdRB_R, overlapRB);
    cv::meanStdDev(B_YCbCr, meanRB_B, stdRB_B, overlapRB);
    cv::meanStdDev(B_YCbCr, meanBL_B, stdBL_B, overlapBL);
    cv::meanStdDev(L_YCbCr, meanBL_L, stdBL_L, overlapBL);
    cv::meanStdDev(L_YCbCr, meanLF_L, stdLF_L, overlapLF);
    cv::meanStdDev(F_YCbCr, meanLF_F, stdLF_F, overlapLF);

    for(int iChannel = 0; iChannel<3; iChannel++){
        Eigen::Matrix4d iA;
        iA << 	stdFR_F[iChannel] , -stdFR_R[iChannel] ,	 0.0  			,	 	0.0		,
                0.0 		 , stdRB_R[iChannel]  , 	 -stdRB_B[iChannel] 	,		0.0		,
                0.0			 , 0.0			 ,   stdBL_B[iChannel] 	,  -stdBL_L[iChannel],
                -stdLF_L[iChannel], 0.0			 , 	0.0				,  stdLF_F[iChannel] ;
        Eigen::EigenSolver<Eigen::Matrix4d> solver(iA);
        Eigen::Vector4d values = solver.eigenvalues().real();
        Eigen::Matrix4d vectors = solver.eigenvectors().real();
        double minValue = values[0];
        int minPos = 0;
        for (int i=1;i<4;i++){
            if (values[i] < minValue){
                minValue = values[i];
                minPos = i;
            }
        }
        Eigen::Vector4d iNewG(1.0 , 1.0 , 1.0 , 1.0);
        iNewG = vectors.block<4,1>(0 , minPos);
//        double min = iNewG[0];
//        for (int i=1;i<4;i++){
//            if (min * min < iNewG[i] * iNewG[i]){
//                min = iNewG[i];
//            }
//        }
        double mean = iNewG.mean();
//        std::cout<<iNewG<<std::endl;
//        std::cout<<min<<std::endl;
        iNewG /= mean;
//        std::cout<<iNewG<<std::endl;

        Eigen::Matrix4d iA2;
        iA2 << 	1 , -1 ,	 0.0  			,	 	0.0		,
                0.0 		 , 1  , 	 -1 	,		0.0		,
                0.0			 , 0.0			 ,   1	,  -1,
                -1, 0.0			 , 	0.0				,  1 ;
        Eigen::Vector4d ib2;
        ib2 << -meanFR_F[iChannel]*iNewG[0] + meanFR_R[iChannel]*iNewG[1],
                -meanRB_R[iChannel]*iNewG[1] + meanRB_B[iChannel]*iNewG[2],
                -meanBL_B[iChannel]*iNewG[2] + meanBL_L[iChannel]*iNewG[3],
                -meanLF_L[iChannel]*iNewG[3] + meanLF_F[iChannel]*iNewG[0];

        Eigen::Vector4d b = iA2.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(ib2);

//        std::cout<<b<<std::endl;

        F_channels[iChannel] = F_channels[iChannel]*iNewG[0] + b[0];
        R_channels[iChannel] = R_channels[iChannel]*iNewG[1] + b[1];
        B_channels[iChannel] = B_channels[iChannel]*iNewG[2] + b[2];
        L_channels[iChannel] = L_channels[iChannel]*iNewG[3] + b[3];

    }

    cv::merge(F_channels, F_YCbCr);
    cv::merge(L_channels, L_YCbCr);
    cv::merge(B_channels, B_YCbCr);
    cv::merge(R_channels, R_YCbCr);

    cv::cvtColor(F_YCbCr, imgF, cv::COLOR_YCrCb2BGR);
    cv::cvtColor(L_YCbCr, imgL, cv::COLOR_YCrCb2BGR);
    cv::cvtColor(B_YCbCr, imgB, cv::COLOR_YCrCb2BGR);
    cv::cvtColor(R_YCbCr, imgR, cv::COLOR_YCrCb2BGR);
}

cv::Scalar FR_mean({0,0,0,0});
cv::Scalar RB_mean({0,0,0,0});
cv::Scalar BL_mean({0,0,0,0});
cv::Scalar LF_mean({0,0,0,0});

cv::Scalar scalarabs(cv::Scalar sc){
    cv::Scalar res(sc);
    for(int i=0; i<3; i++){
        res[i] = std::abs(sc[i]);
    }
    return res;
}

void pc::Sticher::compute_photometricAlignment_std(cv::UMat &imgF, cv::UMat &imgR, cv::UMat &imgB, cv::UMat &imgL){
    cv::Mat F_YCbCr, L_YCbCr, B_YCbCr, R_YCbCr;
    cv::cvtColor(imgF, F_YCbCr, cv::COLOR_BGR2YCrCb);
    cv::cvtColor(imgR, R_YCbCr, cv::COLOR_BGR2YCrCb);
    cv::cvtColor(imgB, B_YCbCr, cv::COLOR_BGR2YCrCb);
    cv::cvtColor(imgL, L_YCbCr, cv::COLOR_BGR2YCrCb);
    cv::Scalar meanFR_F, meanFR_R, meanRB_R, meanRB_B, meanBL_B, meanBL_L, meanLF_L, meanLF_F;
    cv::Scalar stdFR_F, stdFR_R, stdRB_R, stdRB_B, stdBL_B, stdBL_L, stdLF_L, stdLF_F;
    cv::meanStdDev(F_YCbCr, meanFR_F, stdFR_F, overlapFR);
    cv::meanStdDev(R_YCbCr, meanFR_R, stdFR_R, overlapFR);
    cv::meanStdDev(R_YCbCr, meanRB_R, stdRB_R, overlapRB);
    cv::meanStdDev(B_YCbCr, meanRB_B, stdRB_B, overlapRB);
    cv::meanStdDev(B_YCbCr, meanBL_B, stdBL_B, overlapBL);
    cv::meanStdDev(L_YCbCr, meanBL_L, stdBL_L, overlapBL);
    cv::meanStdDev(L_YCbCr, meanLF_L, stdLF_L, overlapLF);
    cv::meanStdDev(F_YCbCr, meanLF_F, stdLF_F, overlapLF);

    FR_mean += scalarabs(meanFR_F-meanFR_R);
    RB_mean += scalarabs(meanRB_R-meanRB_B);
    BL_mean += scalarabs(meanBL_B-meanBL_L);
    LF_mean += scalarabs(meanLF_L-meanLF_F);

}


void pc::Sticher::stich(cv::Mat& result, const std::vector<cv::Mat>& imgs, std::vector<cv::UMat>& remapImgs, int blend_type){
    // remap
    std::vector<cv::UMat> images_wraped_f(numCamera);
    std::vector<cv::UMat> resize_images_wraped_f(numCamera);

    for(int i=0;i<pc::numCamera;i++) {
        cv::remap(imgs[i], remapImgs[i], xMaps[i], yMaps[i], cv::INTER_LINEAR, cv::BORDER_REFLECT);
    }


    photometricAlignment_std(remapImgs[0], remapImgs[1], remapImgs[2], remapImgs[3]);

    // original exposure compensation
//    cv::Ptr<cv::detail::ExposureCompensator> compensator =
//            cv::detail::ExposureCompensator::createDefault(cv::detail::ExposureCompensator::CHANNELS);
//    compensator->feed(corners, remapImgs, maskMaps);    //得到曝光补偿器
//    for(int i=0;i<4;++i)    //应用曝光补偿器，对图像进行曝光补偿
//    {
//        compensator->apply(i, corners[i], remapImgs[i], maskMaps[i]);
//    }

    // resize
    for(int i=0;i<pc::numCamera;i++) {
//        remapImgs[i].convertTo(images_wraped_f[i], CV_32F);
        cv::resize(remapImgs[i], resize_images_wraped_f[i], pc::resizeStitchResultSize);
    }


    if(mySeamFinder->find_dp_temporal_fast(resize_images_wraped_f, resizeMaskMapsSeam, true))
    {
        for(int i=0;i<pc::numCamera;i++) {
            cv::resize(resizeMaskMapsSeam[i], maskMapsSeam[i], pc::stitchResultSize, 0, 0);
        }
    }

    switch(blend_type){
        case cv::detail::Blender::NO:
            blenderPtr = cv::detail::Blender::createDefault( cv::detail::Blender::NO);
            break;

        case cv::detail::Blender::FEATHER:
            blenderPtr = cv::makePtr<cv::detail::FeatherBlender>(0.5);
            break;

        case cv::detail::Blender::MULTI_BAND:
            blenderPtr = cv::makePtr<cv::detail::MultiBandBlender>( false, pc::numBands);
            break;

    }

    blenderPtr->prepare(cv::Rect(0,0,pc::stitchResultSize.width, pc::stitchResultSize.height));

    std::vector<cv::UMat> remapImgs_16(numCamera);

    for(int i=0;i<pc::numCamera;i++){
        remapImgs[i].convertTo(remapImgs_16[i], CV_16SC3);
        blenderPtr->feed(remapImgs_16[i], maskMapsSeam[i], cv::Point(0,0));
    }
    cv::Mat result_s, result_mask;
    blenderPtr->blend(result_s, result_mask);
    result_s.convertTo(result, CV_8U);
}


void pc::Sticher::stich_ori(cv::Mat& result, const std::vector<cv::Mat>& imgs, std::vector<cv::UMat>& remapImgs, int seam_type, int blend_type){
    // remap
    std::vector<cv::UMat> images_wraped_f(numCamera);
    std::vector<cv::UMat> resize_images_wraped_f(numCamera);

    for(int i=0;i<pc::numCamera;i++) {
        cv::remap(imgs[i], remapImgs[i], xMaps[i], yMaps[i], cv::INTER_LINEAR, cv::BORDER_REFLECT);
    }

    photometricAlignment_std(remapImgs[0], remapImgs[1], remapImgs[2], remapImgs[3]);

    // resize
    for(int i=0;i<pc::numCamera;i++) {
        remapImgs[i].convertTo(images_wraped_f[i], CV_32F);
        cv::resize(images_wraped_f[i], resize_images_wraped_f[i], pc::resizeStitchResultSize);
    }

    switch(seam_type){
        case cv::detail::SeamFinder::NO:
            for(int i=0;i<pc::numCamera;i++) {
                maskMaps[i].copyTo(maskMapsSeam[i]);
            }
            break;

        case  cv::detail::SeamFinder::VORONOI_SEAM:
            seamFinderPtr = cv::makePtr<cv::detail::VoronoiSeamFinder>();
            for(int i=0;i<pc::numCamera;i++) {
                maskMaps[i].copyTo(maskMapsSeam[i]);
                cv::resize(maskMapsSeam[i], resizeMaskMapsSeam[i], pc::resizeStitchResultSize);
            }
            seamFinderPtr->find(resize_images_wraped_f, corners, resizeMaskMapsSeam);
            for(int i=0;i<pc::numCamera;i++) {
                cv::resize(resizeMaskMapsSeam[i], maskMapsSeam[i], pc::stitchResultSize);
            }
            break;

        case  cv::detail::SeamFinder::DP_SEAM:
            seamFinderPtr = cv::makePtr<cv::detail::DpSeamFinder>();
            for(int i=0;i<pc::numCamera;i++) {
                maskMaps[i].copyTo(maskMapsSeam[i]);
                cv::resize(maskMapsSeam[i], resizeMaskMapsSeam[i], pc::resizeStitchResultSize);
            }
            seamFinderPtr->find(resize_images_wraped_f, corners, resizeMaskMapsSeam);
            for(int i=0;i<pc::numCamera;i++) {
                cv::resize(resizeMaskMapsSeam[i], maskMapsSeam[i], pc::stitchResultSize);
            }
            break;

        case 3:
            seamFinderPtr = cv::makePtr<cv::detail::GraphCutSeamFinder>(cv::detail::GraphCutSeamFinderBase::COST_COLOR);
            for(int i=0;i<pc::numCamera;i++) {
                maskMaps[i].copyTo(maskMapsSeam[i]);
                cv::resize(maskMapsSeam[i], resizeMaskMapsSeam[i], pc::resizeStitchResultSize);
            }
//            seamFinderPtr->find(images_wraped_f, corners, maskMapsSeam);
            seamFinderPtr->find(resize_images_wraped_f, corners, resizeMaskMapsSeam);

            for(int i=0;i<pc::numCamera;i++) {
                cv::resize(resizeMaskMapsSeam[i], maskMapsSeam[i], pc::stitchResultSize);
            }

            break;
    }


    switch(blend_type){
        case cv::detail::Blender::NO:
            blenderPtr = cv::detail::Blender::createDefault( cv::detail::Blender::NO);
            break;

        case cv::detail::Blender::FEATHER:
            blenderPtr = cv::makePtr<cv::detail::FeatherBlender>(1);
            break;

        case cv::detail::Blender::MULTI_BAND:
            blenderPtr = cv::makePtr<cv::detail::MultiBandBlender>( false, pc::numBands);
            break;

    }

    blenderPtr->prepare(cv::Rect(0,0,pc::stitchResultSize.width, pc::stitchResultSize.height));

    std::vector<cv::UMat> remapImgs_16(numCamera);

    for(int i=0;i<pc::numCamera;i++){
        remapImgs[i].convertTo(remapImgs_16[i], CV_16SC3);
        blenderPtr->feed(remapImgs_16[i], maskMapsSeam[i], cv::Point(0,0));
    }
    cv::Mat result_s, result_mask;
    blenderPtr->blend(result_s, result_mask);
    result_s.convertTo(result, CV_8U);
}