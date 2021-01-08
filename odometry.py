import os
import pickle

import numpy as np

from compare_calibration import read_real_cal_matrix
from extract_keypoints import SerializableKp
from feature_matching import detect, draw_motion, draw_keypoints
from configs import *


def matches2list(kp1: list, kp2: list, matches: list):
    """
    Transforms a list of matches to an array of keypoint where each match is defined by its index.
    :param kp1:                         Set of keypoints in the previous frame.
    :param kp2:                         Set of keypoints in the actual frame.
    :param matches:                     Matches.
    :return: A tuple.
        1. Numpy array with selected matches in the previous frame.
        2. Numpy array with selected matches in the actual frame.
    """
    # list of matches to list of points
    points1, points2 = [None] * len(matches), [None] * len(matches)
    for idx, match in enumerate(matches):
        pt2 = tuple(map(int, kp1[match.trainIdx].pt))
        pt1 = tuple(map(int, kp2[match.queryIdx].pt))
        points1[idx] = np.array(pt1)
        points2[idx] = np.array(pt2)
    return np.array(points1), np.array(points2)


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


class VO:

    """
    Helper Class used to execute the Visual Odometry Pipeline iteratively.
    """
    def __init__(self, video: str, cal_path: str, kp_path: str, matcher_config: dict, video_prefix: str = 'data',
                 kp_prefix: str = 'keypoints'):

        config = StereoSGBMConfig()
        self.left_matcher = cv2.StereoSGBM_create(minDisparity=config.min_disparity,
                                                  numDisparities=config.num_disparities,
                                                  blockSize=config.sad_window_size,
                                                  uniquenessRatio=config.uniqueness_ratio,
                                                  speckleWindowSize=config.speckle_window_size,
                                                  speckleRange=config.speckle_range,
                                                  disp12MaxDiff=config.disp_max_diff,
                                                  preFilterCap=config.pre_filter_cap,
                                                  P1=config.p1,
                                                  P2=config.p2)
        self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.left_matcher)
        self.wls_filter.setLambda(8000.0)
        self.wls_filter.setSigmaColor(1.5)
        self.matcher_config = matcher_config

        # load keypoints
        with open(os.path.join(kp_prefix, kp_path), 'rb') as f:
            self.kps = pickle.load(f)

        # load calibration matrix
        self.cal_path = cal_path

        # used only for 5-point algorithm
        _, self.cal_matrix, self.dist_matrix, _, _ = read_real_cal_matrix(cal_path, n_cam=0)

        # used only for pnp algorithm
        _, self.k1, self.d1, _, _ = read_real_cal_matrix(self.cal_path, n_cam=0)
        s, k2, d2, rotation, translation = read_real_cal_matrix(self.cal_path, n_cam=1)

        # find projections and maps
        R1, R2, P1, P2, self.Q, _, _ = cv2.stereoRectify(self.k1, self.d1, k2, d2, s, rotation, translation)

        # single camera frames directory
        self.left_camera_path = os.path.join(video_prefix, video, 'image_00', 'data')
        self.right_camera_path = os.path.join(video_prefix, video, 'image_01', 'data')

        # init first image and its keypoints, descriptors
        self.pre_img_path = os.path.join(self.left_camera_path, '0000000000.png')
        self.pre_kp = SerializableKp.serializable2cv(self.kps['0000000000.png']['kp'])
        self.pre_des = self.kps['0000000000.png']['des']
        self.actual_idx = 1
        self.last_frame = len(os.listdir(self.left_camera_path))

        # position tracking
        self.pos = np.array([0, 0, 0, 1])[:, np.newaxis]
        self.initial = np.array([0, 0, 0, 1])[:, np.newaxis]
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))

    def step(self, method_match: str = 'hough', method_track: str = 'five-points'):
        """
        Executes one-step prediction of the trajectory using a specific method for matching and a specific
        method for tracking
        :param method_match:                   hough or ransac.
        :param method_track:                   five-points or pnp.
        :return: A tuple:
            1. Actual frame.
            2. Motion frame (contains arrows between each matched pair of points).
            3. The position as a tuple.
            4. If PnP is used return the calculated depth map.
        """
        if self.actual_idx < self.last_frame:
            actual_frame_path = os.path.join(self.left_camera_path, rf'0000000{str(self.actual_idx).zfill(3)}.png')
            actual_frame = cv2.imread(actual_frame_path)
            actual_frame_copy = np.copy(actual_frame)

            # load the keypoints and descriptors
            actual_kp = SerializableKp.serializable2cv(self.kps[os.path.basename(actual_frame_path)]['kp'])
            actual_des = self.kps[os.path.basename(actual_frame_path)]['des']

            # match them
            kp1, kp2, matches = detect(self.pre_kp, actual_kp, self.pre_des, actual_des, method=method_match,
                                       **self.matcher_config)

            # draw motion arrows and keypoints in actual frame
            draw_motion(actual_frame, kp1, kp2, matches)
            draw_keypoints(actual_frame_copy, kp2)

            # estimate motion
            points1, points2 = matches2list(kp1, kp2, matches)
            if method_track == 'five-points':
                self.step_five_points(points1, points2)
                depth_map = None

            elif method_track == 'pnp':
                depth_map = self.step_pnp(points1, points2)

            else:
                raise Exception(f"Method {method_track} not allowed.")

            T = np.vstack([np.hstack([self.R, self.t]), [0, 0, 0, 1]])
            self.pos = T @ self.initial
            # self.pos = self.t

            x, y = self.pos[0][0], self.pos[2][0]
            print(f"Position: {x:.3f}, {y:.3f}")

            self.pre_kp = actual_kp
            self.pre_des = actual_des
            self.pre_img_path = actual_frame_path
            self.actual_idx += 1

            # return raw frame, motion frame, and predicted camera position
            return actual_frame_copy, actual_frame, (x, y), depth_map
        else:
            return None, None, None, None

    def step_pnp(self, points1, points2):
        """
        Finds rotation and translation matrix using Perspectiv-n-Point algorithm from a set of matched points.
        :param points1:                 Set of keypoints.
        :param points2:                 Another set of keypoints which matches the first one.
        :return:                        A suitable representation of the calculated depth map.
        """

        # load left and right frame
        l_frame = cv2.imread(self.pre_img_path)
        pre_right_img_path = os.path.join(self.right_camera_path,
                                          rf'0000000{str(self.actual_idx - 1).zfill(3)}.png')
        r_frame = cv2.imread(pre_right_img_path)

        # disparity map creation
        left_disp = self.left_matcher.compute(l_frame, r_frame) / 16.
        left_disp = left_disp.astype(np.float32)
        # right_disp = self.right_matcher.compute(r_frame, l_frame)
        # result = self.wls_filter.filter(left_disp, l_frame, None, right_disp)
        points_3d = cv2.reprojectImageTo3D(left_disp, self.Q, handleMissingValues=True)

        # select valid keypoints
        p3d_valid, mask = select_points(points_3d, points1)
        p2d = points2[mask]
        p3d_valid = p3d_valid.astype(np.float32)
        p2d = p2d.astype(np.float32)

        # find rotation and translation  matrix
        ret, R, t, _ = cv2.solvePnPRansac(p3d_valid, p2d, self.k1, None)
        # R, t = cv2.solvePnPRefineLM(p3d_valid, p2d, self.k1, None, R, t)
        R, _ = cv2.Rodrigues(R, None)
        if not ret:
            print("wrong pred")

        self.R = R @ self.R
        self.t = self.t + self.R @ t
        return cv2.normalize(left_disp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    def step_five_points(self, points1, points2):
        """
        Finds rotation and translation matrix using Five-Point algorithm from a set of matched points.
        :param points1:                 Set of keypoints.
        :param points2:                 Another set of keypoints which matches the first one.
        """
        E, mask = cv2.findEssentialMat(points1, points2, self.cal_matrix, method=cv2.RANSAC,
                                       prob=0.999, threshold=1.0)
        _, R, t, _ = cv2.recoverPose(E, points1, points2, cameraMatrix=self.cal_matrix, mask=mask)

        # scale = self.get_scale(f'data/odometry/poses/07.txt')

        self.R = R @ self.R
        self.t += self.R @ t

    # def get_scale(self, gt_path: str):
    #     # ground_truth = f'data/odometry/poses/{ground_truths[date][id]}.txt'
    #     with open(gt_path, 'r') as f:
    #         lines = f.readlines()
    #     pose = lines[self.actual_idx-1].strip().split()
    #     x_prev = float(pose[3])
    #     y_prev = float(pose[7])
    #     z_prev = float(pose[11])
    #     pose = lines[self.actual_idx].strip().split()
    #     x = float(pose[3])
    #     y = float(pose[7])
    #     z = float(pose[11])
    #
    #     return np.sqrt((x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + (z - z_prev)*(z - z_prev))
