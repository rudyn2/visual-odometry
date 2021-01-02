import os
from extract_keypoints import SerializableKp
from feature_matching import *

import cv2
import numpy as np
from configs import *
from compare_calibration import read_real_cal_matrix
import pickle
from feature_matching import detect
from configs import MatcherConfig
from odometry import filter_kp


def select_points(p3d, p2d):
    """
    Selects 3D points from a 3D image using a 2D array.
    :param p3d:                 3d image (W, H, 3). Last channel is associated with z-dimension.
    :param p2d:                 2d array points (N, 2). Last channel is associated with image dimension.
    :return:                    Points selected (M, 3), mask
    """
    p3d_selected = np.array([p3d[p2d[idx, 1], p2d[idx, 0]] for idx in range(p2d.shape[0])])
    valid_mask = ~np.isinf(p3d_selected).any(axis=1)
    return p3d_selected[valid_mask], valid_mask


if __name__ == '__main__':
    date = '2011_09_26'
    id = '0001'
    video = f'{date}/{date}_drive_{id}_sync'
    keypoints = f'{date}/{date}_drive_{id}_sync/image_00/sift_keypoints.pkl'
    experimental_cal_path = f'data/{date}/calib_cam_to_cam.txt'

    # load keypoints
    with open(os.path.join('keypoints', keypoints), 'rb') as f:
        kps = pickle.load(f)

    _, k1, d1, _, _ = read_real_cal_matrix(experimental_cal_path, n_cam=0)
    s, k2, d2, rotation, translation = read_real_cal_matrix(experimental_cal_path, n_cam=1)
    # find projections and maps
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(k1, d1, k2, d2, s, rotation, translation)

    sgbm_config = StereoSGBMConfig()
    left_matcher = cv2.StereoSGBM_create(minDisparity=sgbm_config.min_disparity,
                                         numDisparities=sgbm_config.num_disparities,
                                         blockSize=sgbm_config.sad_window_size,
                                         preFilterCap=sgbm_config.pre_filter_cap,
                                         P1=sgbm_config.p1,  # 8*3*win_size**2,
                                         P2=sgbm_config.p2,
                                         mode=sgbm_config.mode,
                                         )

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    wls_filter.setLambda(8000.0)
    wls_filter.setSigmaColor(1.5)

    left_camera_path = os.path.join('data', video, 'image_00', 'data')
    right_camera_path = os.path.join('data', video, 'image_01', 'data')

    # init first image and its keypoints, descriptors
    pre_img_path = os.path.join(left_camera_path, '0000000000.png')
    pre_kp = SerializableKp.serializable2cv(kps['0000000000.png']['kp'])
    pre_des = kps['0000000000.png']['des']

    apply_filtering = True
    for i in range(1, 100):

        # load the keypoints and descriptors
        actual_kp = SerializableKp.serializable2cv(kps[os.path.basename(os.path.join(left_camera_path, f'0000000{str(i).zfill(3)}.png'))]['kp'])
        actual_des = kps[os.path.basename(os.path.join(left_camera_path, f'0000000{str(i).zfill(3)}.png'))]['des']

        kp1, kp2, matches = detect(pre_kp, actual_kp, pre_des, actual_des, method='hough', **MatcherConfig.hough)
        points1, points2 = filter_kp(kp1, kp2, matches)

        # load left and right frame
        l_frame = cv2.imread(os.path.join(left_camera_path, f'0000000{str(i).zfill(3)}.png'))
        r_frame = cv2.imread(os.path.join(right_camera_path, f'0000000{str(i).zfill(3)}.png'))

        # compute points
        left_for_matcher = np.copy(l_frame)
        right_for_matcher = np.copy(r_frame)

        # disparity map creation
        left_disp = left_matcher.compute(left_for_matcher, right_for_matcher)
        right_disp = right_matcher.compute(right_for_matcher, left_for_matcher)

        if apply_filtering:
            result = wls_filter.filter(left_disp, l_frame, None, right_disp)
        else:
            result = left_disp

        points_3d = cv2.reprojectImageTo3D(result, Q, handleMissingValues=True)
        depth_map = cv2.normalize(left_disp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # draw motion arrows in actual frame
        # select valid keypoints
        p3d_valid, mask = select_points(points_3d, points1)
        p2d = points2[mask]
        p3d_valid = p3d_valid.astype(np.float32)
        p2d = p2d.astype(np.float32)
        ret, R, t, _ = cv2.solvePnPRansac(p3d_valid, p2d, k1, d1)

        draw_motion(points_3d, kp1, kp2, matches, color=(255, 255, 255))

        cv2.imshow('Left frame', l_frame)
        cv2.imshow('Depth map', depth_map)
        cv2.imshow('3D Points', points_3d)
        cv2.waitKey(50)

        pre_kp = actual_kp
        pre_des = actual_des

    cv2.destroyAllWindows()
