import cv2
from compare_calibration import read_real_cal_matrix
import os
import os

import cv2

from compare_calibration import read_real_cal_matrix

if __name__ == '__main__':
    date = '2011_09_26'
    id = '0001'
    video = f'{date}/{date}_drive_{id}_sync'
    keypoints = f'{date}/{date}_drive_{id}_sync/image_00/sift_keypoints.pkl'
    experimental_cal_path = f'data/{date}/calib_cam_to_cam.txt'

    _, k1, d1, _, _ = read_real_cal_matrix(experimental_cal_path, n_cam=0)
    s, k2, d2, rotation, translation = read_real_cal_matrix(experimental_cal_path, n_cam=1)

    # find projections and maps
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(k1, d1, k2, d2, s, rotation, translation)
    l_transform1, l_transform2 = cv2.initUndistortRectifyMap(k1, d1, R1, P1, s, cv2.CV_32FC1)
    r_transform1, r_transform2 = cv2.initUndistortRectifyMap(k2, d2, R2, P2, s, cv2.CV_32FC1)

    # load left and right frame
    l_frame = cv2.imread(os.path.join('data', video, 'image_00', 'data', '0000000000.png'))
    r_frame = cv2.imread(os.path.join('data', video, 'image_01', 'data', '0000000000.png'))

    # apply maps to each frame
    l_frame_map = cv2.remap(l_frame, l_transform1, l_transform2, None)
    r_frame_map = cv2.remap(r_frame, r_transform1, r_transform2, None)

    # compute points
    pre_filter_cap = 63
    sad_window_size = 3
    p1 = sad_window_size * sad_window_size * 4
    p2 = sad_window_size * sad_window_size * 32
    min_disparity = 0
    num_disparities = 128
    uniqueness_ratio = 10
    speckle_window_size = 100
    speckle_range = 32
    disp_max_diff = 1
    full_dp = 1
    stereo = cv2.StereoSGBM_create(minDisparity=min_disparity,
                                   numDisparities=num_disparities,
                                   blockSize=7,
                                   uniquenessRatio=uniqueness_ratio,
                                   speckleWindowSize=speckle_window_size,
                                   speckleRange=speckle_range,
                                   disp12MaxDiff=disp_max_diff,
                                   preFilterCap=pre_filter_cap,
                                   P1=p1,  # 8*3*win_size**2,
                                   P2=p2)  # 32*3*win_size**2)
    disparity_map = stereo.compute(l_frame, r_frame)
    disparity_map = cv2.normalize(disparity_map, disparity_map, alpha=0, beta=255,
                                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # disparity_map = cv2.GaussianBlur(disparity_map, ksize=(3, 3), sigmaX=2)
    points_3d = cv2.reprojectImageTo3D(disparity_map, Q, handleMissingValues=True)

    cv2.imshow('Left frame', l_frame)
    cv2.imshow('Disparity map', disparity_map)
    cv2.imshow('3D Points', points_3d)
    cv2.waitKey()
    cv2.destroyAllWindows()
