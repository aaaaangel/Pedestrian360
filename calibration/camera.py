import numpy as np
import cv2
import os


class CalibrateFrame:
    def __init__(self, file_path):
        self.file_path = file_path
        self.img = cv2.imread(file_path)
        if self.img is None:
            raise Exception("The calibrate frame"+file_path+"does not exist!")
        self.pattern_size = 0
        self.corners = []
        self.undistort_corners = []

    def extractBoardPoints(self, K, D):
        is_found, self.corners = cv2.findChessboardCorners(self.img, self.pattern_size)
        if not is_found:
            raise Exception("The chessboard cannot be detected in this frame" + self.file_path)
        self.undistort_corners = cv2.fisheye.undistortPoints(self.corners, K, D, P=K)


class Camera:
    def __init__(self):
        # extrinsics
        self.R = np.identity(3)
        self.t = np.zeros((3, 1))

        # intrinsics
        self.K = np.zeros((3, 3))
        self.D = np.zeros((4, 1))
        self.frames1 = []
        self.frames2 = []

    def setK(self, fx, fy, cx, cy):
        self.K[0, 0] = fx
        self.K[1, 1] = fy
        self.K[0, 2] = cx
        self.K[1, 2] = cy
        self.K[2, 2] = 1

    def setD(self, k1, k2, k3, k4):
        self.D[0, 0] = k1
        self.D[1, 0] = k2
        self.D[2, 0] = k3
        self.D[3, 0] = k4

    def setFrames1(self, folder_path):
        file_names = os.listdir(folder_path)
        for file_name in file_names:
            if file_name.endswith('.DS_Store') or file_name.startswith('.'):
                continue
            file_path = os.path.join(folder_path, file_name)
            self.frames1.append(CalibrateFrame(file_path))

    def setFrames2(self, folder_path):
        file_names = os.listdir(folder_path)
        for file_name in file_names:
            if file_name.endswith('.DS_Store') or file_name.startswith('.'):
                continue
            file_path = os.path.join(folder_path, file_name)
            self.frames2.append(CalibrateFrame(file_path))

    def calibrateAllFrames(self, pattern_size=(9, 6)):
        for frame in self.frames1:
            frame.pattern_size = pattern_size
            frame.extractBoardPoints(self.K, self.D)
        for frame in self.frames2:
            frame.pattern_size = pattern_size
            frame.extractBoardPoints(self.K, self.D)

    def computeReprojectionError(self, g3dpoints1, g3dpoints2):
        g3dpoints1_arr = np.concatenate([it.Point3d.reshape((1, 1, 3)) for it in g3dpoints1], axis=0)
        g3dpoints2_arr = np.concatenate([it.Point3d.reshape((1, 1, 3)) for it in g3dpoints2], axis=0)

        Rvec, _ = cv2.Rodrigues(self.R)
        reprojectPoints1, _ = cv2.fisheye.projectPoints(g3dpoints1_arr, Rvec, self.t, self.K, self.D)
        undistort_reprojectPoints1 = cv2.fisheye.undistortPoints(reprojectPoints1, self.K, self.D, P=self.K)
        reprojectPoints2, _ = cv2.fisheye.projectPoints(g3dpoints2_arr, Rvec, self.t, self.K, self.D)
        undistort_reprojectPoints2 = cv2.fisheye.undistortPoints(reprojectPoints2, self.K, self.D, P=self.K)

        sum_error = 0
        if len(self.frames1)>0:
            for i in range(len(self.frames1[0].undistort_corners)):
                sum_error += np.linalg.norm(undistort_reprojectPoints1[i]-self.frames1[0].undistort_corners[i])
        if len(self.frames2)>0:
            for i in range(len(self.frames2[0].undistort_corners)):
                sum_error += np.linalg.norm(undistort_reprojectPoints2[i]-self.frames2[0].undistort_corners[i])
        if len(self.frames1)>0 and len(self.frames2)>0:
            sum_error /= (len(self.frames1[0].undistort_corners) + len(self.frames2[0].undistort_corners))
        else:
            sum_error /= 54

        return sum_error
