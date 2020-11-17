import cv2
from camera import Camera, CalibrateFrame
import numpy as np


class Point:
    def __init__(self):
        self.Point3d = 0
        self.Point2d1 = 0
        self.Point2d2 = 0
        self.dist = 0

def cmp(x):
    return x.dist


def estimateFisheye2d2dByCamera(camera1, camera2):
    # img_size = np.array([1280, 1080])
    gPoints3d = []

    gPoints2d1 = []
    gPoints2d2 = []
    iCenter = np.array([1080/2, 1280/2])

    for k in range(len(camera1.frames1)):
        gPoints = []
        gSubPoint3d = []
        gSubPoint2d1 = []
        gSubPoint2d2 = []
        for j in range(6):
            for i in range(9):
                point = Point()
                point.Point3d = np.array([i, j, 0.0])
                point.Point2d1 = camera1.frames1[k].corners[j * 9 + i]
                point.Point2d2 = camera2.frames2[k].corners[j * 9 + i]
                point.dist = np.linalg.norm(point.Point2d1-iCenter) + np.linalg.norm(point.Point2d2-iCenter)  #以供选择点进行优化
                gPoints.append(point)

        gPoints.sort(key=cmp)

        for i in range(54):
            gSubPoint3d.append(gPoints[i].Point3d)
            gSubPoint2d1.append(gPoints[i].Point2d1)
            gSubPoint2d2.append(gPoints[i].Point2d2)

        gPoints3d.append(np.stack(gSubPoint3d, axis=0).reshape(1,-1,3))
        gPoints2d1.append(np.stack(gSubPoint2d1, axis=0))
        gPoints2d2.append(np.stack(gSubPoint2d2, axis=0))

    gPoints3d_arr = np.stack(gPoints3d, axis=0).astype(np.float32)
    gPoints2d1_arr = np.swapaxes(np.stack(gPoints2d1, axis=0), 1, 2).astype(np.float32)
    gPoints2d2_arr = np.swapaxes(np.stack(gPoints2d2, axis=0), 1, 2).astype(np.float32)

    # objpoints shape: (<num of calibration images>, 1, <num points in set>, 3)
    # imgpoints_left shape: (<num of calibration images>, 1, <num points in set>, 2)
    # imgpoints_right shape: (<num of calibration images>, 1, <num points in set>, 2)
    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T= \
    cv2.fisheye.stereoCalibrate(gPoints3d_arr, gPoints2d1_arr, gPoints2d2_arr, camera1.K, camera1.D, camera2.K, camera2.D, (1920, 1080))

    camera2.R = np.matmul(R, camera1.R)
    camera2.t = np.matmul(R, camera1.t) + T

    print(retval)


def pix2cam(p, K):
    p = p.reshape(1, 2)
    cam_p = np.zeros((1, 2))
    cam_p[0][0] = (p[0][0] - K[0][2])/K[0][0]
    cam_p[0][1] = (p[0][1] - K[1][2]) / K[1][1]
    return cam_p


def triangulationPairs(camera1, camera2):
    gPoints2d1 = np.zeros((0, 1, 2), dtype=np.float32)
    gPoints2d2 = np.zeros((0, 1, 2), dtype=np.float32)
    for k in range(len(camera1.frames1)):
        gPoints2d1 = np.concatenate([gPoints2d1, camera1.frames1[k].undistort_corners], axis=0)
        gPoints2d2 = np.concatenate([gPoints2d2, camera2.frames2[k].undistort_corners], axis=0)

    gRetinalPoints1 = []
    gRetinalPoints2 = []
    for i in range(gPoints2d1.shape[0]):
        gRetinalPoints1.append(pix2cam(gPoints2d1[i, :, :], camera1.K))
        gRetinalPoints2.append(pix2cam(gPoints2d2[i, :, :], camera2.K))
    gRetinalPoints1 = np.stack(gRetinalPoints1)
    gRetinalPoints2 = np.stack(gRetinalPoints2)

    T1 = np.concatenate([camera1.R, camera1.t], axis=1)
    T2 = np.concatenate([camera2.R, camera2.t], axis=1)

    iPoints3d = cv2.triangulatePoints(T1, T2, gRetinalPoints1, gRetinalPoints2)

    triangulationPairsPoints = []
    for i in range(iPoints3d.shape[1]):
        temp = Point()
        temp3d = iPoints3d[:,i] / iPoints3d[3,i]
        temp.Point3d = temp3d[:3].reshape(1, 3)
        temp.Point2d1 = gPoints2d1[i, :].reshape(1, 2)
        temp.Point2d2 = gPoints2d2[i, :].reshape(1, 2)
        triangulationPairsPoints.append(temp)

    return triangulationPairsPoints

