import cv2
import os
import yaml
import numpy as np
import g2o

from camera import Camera
from calibration import estimateFisheye2d2dByCamera, triangulationPairs
from g2o_optimize import PoseGraphOptimization
from visualization import visualize


if __name__ == '__main__':

    intrinsics_file = 'my_intrinsics.yaml'
    extrinsics_frames_path = './extrinsics_frames/'
    out_parm_file_path = './results/'

    # load intrinsics_file
    f = open(intrinsics_file)
    intrinsics = yaml.load(f)
    camera_F = Camera()
    camera_F.setK(intrinsics['FrontCamera.fx'], intrinsics['FrontCamera.fy'], intrinsics['FrontCamera.cx'], intrinsics['FrontCamera.cy'])
    camera_F.setD(intrinsics['FrontCamera.k1'], intrinsics['FrontCamera.k2'], intrinsics['FrontCamera.k3'], intrinsics['FrontCamera.k4'])
    camera_L = Camera()
    camera_L.setK(intrinsics['LeftCamera.fx'], intrinsics['LeftCamera.fy'], intrinsics['LeftCamera.cx'], intrinsics['LeftCamera.cy'])
    camera_L.setD(intrinsics['LeftCamera.k1'], intrinsics['LeftCamera.k2'], intrinsics['LeftCamera.k3'], intrinsics['LeftCamera.k4'])
    camera_B = Camera()
    camera_B.setK(intrinsics['BackCamera.fx'], intrinsics['BackCamera.fy'], intrinsics['BackCamera.cx'], intrinsics['BackCamera.cy'])
    camera_B.setD(intrinsics['BackCamera.k1'], intrinsics['BackCamera.k2'], intrinsics['BackCamera.k3'], intrinsics['BackCamera.k4'])
    camera_R = Camera()
    camera_R.setK(intrinsics['RightCamera.fx'], intrinsics['RightCamera.fy'], intrinsics['RightCamera.cx'], intrinsics['RightCamera.cy'])
    camera_R.setD(intrinsics['RightCamera.k1'], intrinsics['RightCamera.k2'], intrinsics['RightCamera.k3'], intrinsics['RightCamera.k4'])
    camera_F2 = Camera()
    camera_F2.setK(intrinsics['FrontCamera.fx'], intrinsics['FrontCamera.fy'], intrinsics['FrontCamera.cx'], intrinsics['FrontCamera.cy'])
    camera_F2.setD(intrinsics['FrontCamera.k1'], intrinsics['FrontCamera.k2'], intrinsics['FrontCamera.k3'], intrinsics['FrontCamera.k4'])

    # load frames to cameras
    LFPath_F = extrinsics_frames_path + "LF/LF_F"
    LFPath_L = extrinsics_frames_path + "LF/LF_L"
    BLPath_L = extrinsics_frames_path + "BL/BL_L"
    BLPath_B = extrinsics_frames_path + "BL/BL_B"
    RBPath_B = extrinsics_frames_path + "RB/RB_B"
    RBPath_R = extrinsics_frames_path + "RB/RB_R"
    FRPath_R = extrinsics_frames_path + "FR/FR_R"
    FRPath_F = extrinsics_frames_path + "FR/FR_F"
    camera_F.setFrames1(LFPath_F)
    camera_L.setFrames1(BLPath_L)
    camera_L.setFrames2(LFPath_L)
    camera_B.setFrames1(RBPath_B)
    camera_B.setFrames2(BLPath_B)
    camera_R.setFrames1(FRPath_R)
    camera_R.setFrames2(RBPath_R)
    camera_F2.setFrames2(FRPath_F)

    # ExtractBoardPoints
    camera_F.calibrateAllFrames()
    camera_L.calibrateAllFrames()
    camera_B.calibrateAllFrames()
    camera_R.calibrateAllFrames()
    camera_F2.calibrateAllFrames()

    # Initial estimation
    estimateFisheye2d2dByCamera(camera_F, camera_L)
    estimateFisheye2d2dByCamera(camera_L, camera_B)
    estimateFisheye2d2dByCamera(camera_B, camera_R)
    estimateFisheye2d2dByCamera(camera_R, camera_F2)

    print('==============Initialization in Front coordinate=================')
    print(camera_F.R)
    print(camera_F.t)
    print(camera_L.R)
    print(camera_L.t)
    print(camera_B.R)
    print(camera_B.t)
    print(camera_R.R)
    print(camera_R.t)
    print(camera_F2.R)
    print(camera_F2.t)

    visualize(camera_L.R, camera_L.t, camera_B.R, camera_B.t, camera_R.R, camera_R.t, camera_F2.R, camera_F2.t, 'old')

    # Triangulation
    triangulationPairsPoints_LF = triangulationPairs(camera_F, camera_L)
    triangulationPairsPoints_BL = triangulationPairs(camera_L, camera_B)
    triangulationPairsPoints_RB = triangulationPairs(camera_B, camera_R)
    triangulationPairsPoints_FR = triangulationPairs(camera_R, camera_F2)

    F_err = camera_F.computeReprojectionError(triangulationPairsPoints_LF, triangulationPairsPoints_FR)
    L_err = camera_L.computeReprojectionError(triangulationPairsPoints_BL, triangulationPairsPoints_LF)
    B_err = camera_B.computeReprojectionError(triangulationPairsPoints_RB, triangulationPairsPoints_BL)
    R_err = camera_R.computeReprojectionError(triangulationPairsPoints_FR, triangulationPairsPoints_RB)
    F2_err = camera_F2.computeReprojectionError(triangulationPairsPoints_LF, triangulationPairsPoints_FR)

    print(F_err)
    print(L_err)
    print(B_err)
    print(R_err)
    print(F2_err)

    # Optimiaztion
    poseOptimizer = PoseGraphOptimization()
    # Add pose vertices & Set camera parameter
    pose = g2o.SE3Quat(camera_F.R, camera_F.t.flatten())
    vertexF = poseOptimizer.add_vertex_pose(0, pose, fixed=True)
    parm = g2o.CameraParameters(camera_F.K[0, 0], camera_F.K[0:2, 2], 0)
    parm.set_id(0)
    poseOptimizer.add_parameter(parm)

    vertexes = []
    vertexes.append(vertexF)
    for cam, id in zip([camera_L, camera_B, camera_R, camera_F2], list(range(1, 5))):
        pose = g2o.SE3Quat(cam.R, cam.t.flatten())
        vertex = poseOptimizer.add_vertex_pose(id, pose, fixed=False)
        vertexes.append(vertex)
        parm = g2o.CameraParameters(cam.K[0, 0], cam.K[0:2, 2], 0)
        parm.set_id(id)
        poseOptimizer.add_parameter(parm)

    # Add 3d points and 2 edge from 2d points
    index = 5
    index_points = []
    index_points.append(index)
    index_edge = 5
    for triangulationPairsPoint in triangulationPairsPoints_LF:
        point = g2o.VertexSBAPointXYZ()
        point.set_id(index)
        point.set_estimate(triangulationPairsPoint.Point3d.flatten())
        poseOptimizer.add_vertex(point)

        # Add edge to front camera pose
        kernel = g2o.RobustKernelHuber()
        kernel.set_delta(1)
        poseOptimizer.add_edge(index_edge, [point, vertexes[0]], triangulationPairsPoint.Point2d1.flatten(), 0,
                 robust_kernel=kernel)
        index_edge+=1

        # Add edge to front camera2 pose
        kernel = g2o.RobustKernelHuber()
        kernel.set_delta(1)
        poseOptimizer.add_edge(index_edge, [point, vertexes[4]], triangulationPairsPoint.Point2d1.flatten(), 0,
                               robust_kernel=kernel)
        index_edge += 1

        # Add edge to left camera pose
        kernel = g2o.RobustKernelHuber()
        kernel.set_delta(1)
        poseOptimizer.add_edge(index_edge, [point, vertexes[1]], triangulationPairsPoint.Point2d2.flatten(), 1,
                               robust_kernel=kernel)
        index_edge += 1

        index+=1

    for triangulationPairsPoints_AB, ptr in \
            zip([triangulationPairsPoints_BL, triangulationPairsPoints_RB], [1,2]):
        index_points.append(index)
        for triangulationPairsPoint in triangulationPairsPoints_AB:
            point = g2o.VertexSBAPointXYZ()
            point.set_id(index)
            point.set_estimate(triangulationPairsPoint.Point3d.flatten())
            poseOptimizer.add_vertex(point)

            # Add edge to left camera pose
            kernel = g2o.RobustKernelHuber()
            kernel.set_delta(1)
            poseOptimizer.add_edge(index_edge, [point, vertexes[ptr]], triangulationPairsPoint.Point2d1.flatten(), ptr,
                                   robust_kernel=kernel)
            index_edge += 1

            # Add edge to back camera pose
            kernel = g2o.RobustKernelHuber()
            kernel.set_delta(1)
            poseOptimizer.add_edge(index_edge, [point, vertexes[ptr+1]], triangulationPairsPoint.Point2d2.flatten(), ptr+1,
                                   robust_kernel=kernel)
            index_edge += 1

            index += 1

    index_points.append(index)
    for triangulationPairsPoint in triangulationPairsPoints_FR:
        point = g2o.VertexSBAPointXYZ()
        point.set_id(index)
        point.set_estimate(triangulationPairsPoint.Point3d.flatten())
        poseOptimizer.add_vertex(point)

        # Add edge to right camera pose
        kernel = g2o.RobustKernelHuber()
        kernel.set_delta(1)
        poseOptimizer.add_edge(index_edge, [point, vertexes[3]], triangulationPairsPoint.Point2d1.flatten(), 3,
                 robust_kernel=kernel)
        index_edge+=1

        # Add edge to front camera2 pose
        kernel = g2o.RobustKernelHuber()
        kernel.set_delta(1)
        poseOptimizer.add_edge(index_edge, [point, vertexes[4]], triangulationPairsPoint.Point2d2.flatten(), 0,
                               robust_kernel=kernel)
        index_edge += 1

        # Add edge to front camera pose
        kernel = g2o.RobustKernelHuber()
        kernel.set_delta(1)
        poseOptimizer.add_edge(index_edge, [point, vertexes[0]], triangulationPairsPoint.Point2d2.flatten(), 0,
                               robust_kernel=kernel)
        index_edge += 1

        index+=1


    # poseOptimizer.initialize_optimization()
    poseOptimizer.optimize(max_iterations=10000)

    for cam, ind in zip([camera_L, camera_B, camera_R, camera_F2], [1,2,3,4]):
        updateT = vertexes[ind].estimate().matrix()
        cam.R = updateT[0:3, 0:3]
        cam.t = updateT[0:3, 3].reshape(3,1)

    print('==============new in Front coordinate=================')
    print(camera_F.R)
    print(camera_F.t)
    print(camera_L.R)
    print(camera_L.t)
    print(camera_B.R)
    print(camera_B.t)
    print(camera_R.R)
    print(camera_R.t)
    print(camera_F2.R)
    print(camera_F2.t)

    visualize(camera_L.R, camera_L.t, camera_B.R, camera_B.t, camera_R.R, camera_R.t, camera_F2.R, camera_F2.t, 'new')

    # Triangulation
    triangulationPairsPoints_LF = triangulationPairs(camera_F, camera_L)
    triangulationPairsPoints_BL = triangulationPairs(camera_L, camera_B)
    triangulationPairsPoints_RB = triangulationPairs(camera_B, camera_R)
    triangulationPairsPoints_FR = triangulationPairs(camera_R, camera_F2)

    F_err = camera_F.computeReprojectionError(triangulationPairsPoints_LF, triangulationPairsPoints_FR)
    L_err = camera_L.computeReprojectionError(triangulationPairsPoints_BL, triangulationPairsPoints_LF)
    B_err = camera_B.computeReprojectionError(triangulationPairsPoints_RB, triangulationPairsPoints_BL)
    R_err = camera_R.computeReprojectionError(triangulationPairsPoints_FR, triangulationPairsPoints_RB)
    F2_err = camera_F2.computeReprojectionError(triangulationPairsPoints_LF, triangulationPairsPoints_FR)

    print(F_err)
    print(L_err)
    print(B_err)
    print(R_err)
    print(F2_err)


    # center = -(camera_F.t + camera_L.t + camera_B.t + camera_R.t)/4
    center = np.zeros([3, 1])
    for cam in camera_F, camera_L, camera_B, camera_R:
        temp_T = np.concatenate([cam.R, cam.t], axis=1)
        temp_T = np.concatenate([temp_T, np.array([[0,0,0,1]])], axis=0)
        T_inv = np.linalg.inv(temp_T)
        t_F = np.matmul(T_inv, np.array([[0], [0], [0], [1]]))
        center = center + t_F[:3]
    center = center/4


    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1,0]])
    TFG = np.concatenate([R, center], axis=1)
    TFG = np.concatenate([TFG,  np.array([[0,0,0,1]])], axis=0)
    TG_list = []
    for cam in camera_F, camera_L, camera_B, camera_R:
        temp_T = np.concatenate([cam.R, cam.t], axis=1)
        temp_T = np.concatenate([temp_T, np.array([[0,0,0,1]])], axis=0)
        TG = np.matmul(temp_T, TFG)
        TG_list.append(TG)


    print('==============new in Ground coordinate=================')
    print(TG_list[0])
    print(TG_list[1])
    print(TG_list[2])
    print(TG_list[3])

    # Write result file
    name = ['F', 'L', 'B', 'R']
    if not os.path.isdir(out_parm_file_path):
        os.makedirs(out_parm_file_path)
    for id, cam in enumerate([camera_F, camera_L, camera_B, camera_R]):
        temp_T = str(TG_list[id][:3, :]).replace('[','').replace(']','') + '\n'
        K = str(cam.K).replace('[','').replace(']','') + '\n'
        D = str(cam.D.reshape((4, 1))).replace('[','').replace(']','')
        f = open(out_parm_file_path + name[id]+'AffineLX.txt', 'w')
        f.write(temp_T+K+D)
        f.close()











