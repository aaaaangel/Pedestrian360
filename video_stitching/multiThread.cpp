#include <iostream>
#include <string>
#include <ctime>
#include <thread>
#include <semaphore.h>
#include <fcntl.h>
#include <zconf.h>

#include <opencv2/videoio.hpp>

#include "CameraParameter.h"
#include "Projector.h"
#include "Sticher.h"
#include "Consts.h"

cv::VideoWriter writer;
cv::Mat panoramaResult;
sem_t* semaphore_display;

void stich_frame(int tid){
    cv::VideoCapture cap1, cap2, cap3, cap4;   // F, R, B, L
    cap1.open("../resources/videos/F500_0903_0.avi");
    cap2.open("../resources/videos/R500_0903_0.avi");
    cap3.open("../resources/videos/B500_0903_0.avi");
    cap4.open("../resources/videos/L500_0903_0.avi");

    pc::Sticher sticher;
    std::vector<cv::Mat> imgs(pc::numCamera);
    std::vector<cv::UMat> remapImgs;
    for(int i=0;i<pc::numCamera;i++) {
        remapImgs.push_back(cv::UMat(pc::stitchResultSize, CV_8U));
    }
    cv::Mat temp_panoramaResult;

    int temp=200;
    while(temp--){
        cap1>>imgs[0];
        cap2>>imgs[1];
        cap3>>imgs[2];
        cap4>>imgs[3];

        sticher.stich(panoramaResult, imgs, remapImgs);
        sem_post(semaphore_display);
        clock_t time_end2=std::clock();
    }
    return;
}

int main() {

    semaphore_display = sem_open("sem2",O_CREAT, S_IRUSR | S_IWUSR, 0);

    writer.open("../result.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 5, pc::stitchResultSize, true);

    std::thread threads[pc::threads_num];
    for(int i=0; i<pc::threads_num; i++){
        threads[i] = std::thread(std::ref(stich_frame), i);
    }

    int temp=200;
    while(temp--){
        sem_wait(semaphore_display);
        cv::imshow("result", panoramaResult);
        int key = cv::waitKey(5);
        if(key>0)
            break;
        writer.write(panoramaResult);
    }

    writer.release();

    sem_close(semaphore_display);
    sem_unlink("sem1");
}