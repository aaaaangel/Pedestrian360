# Pedestrian360
The implementation of "Pedestrian-Aware Panoramic Video Stitching Based on a Structured Camera Aray".



## Instructions

* **calibration** : `calibration/calibrate_extrinsics`:  calibrate the extrinsics of the multi-camera system through the images in `calibration/extrinsics_frames`. The results are saved in `calibration/results`
* **video_stitching**: Run `video_stitching/multiThread.cpp` to stitch the videos. 
* **server**: `server/server.py` receive the frames from the stitching program and perform human segmentation using Mask R-CNN. Then it will send the mask data to the stitching program.
* If you want to speed up the stitching, set the rescale factor in `video_stitching/const.h` and `server/server.py` at the same time.



## Requirements

* Python 3.6
  * torch 1.3.0
  * torchvision 0.4.1
  * numpy
  * g2o
  * yaml
  * cv2
  * OpencGL
  * pangolin
* C++
  * cmake
  * OpenCV 4.2.0
  * eigen3
  * Threads

