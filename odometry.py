import os
import pickle

import cv2
import numpy as np

from compare_calibration import read_real_cal_matrix
from extract_keypoints import SerializableKp
from feature_matching import detect, draw_motion


def process_kp(kp_list: list):
    points = []
    for kp in kp_list:
        points.append(list(kp.pt))
    return np.array(points)


def filter_kp(kp1: list, kp2: list, matches: list):
    # list of matches to list of points
    points1, points2 = [], []
    for match in matches:
        pt2 = tuple(map(int, kp1[match.trainIdx].pt))
        pt1 = tuple(map(int, kp2[match.queryIdx].pt))
        points1.append(np.array(pt1))
        points2.append(np.array(pt2))
    return np.array(points1), np.array(points2)

class VO:
    def __init__(self, video: str, cal_path: str, kp_path: str, video_prefix: str = 'data',
                 kp_prefix: str = 'keypoints'):

        # parameters extracted from:
        # http://www.cvlibs.net/datasets/kitti/eval_stereo_flow_detail.php?benchmark=stereo&error=3&eval=all&result=3ae300a3a3b3ed3e48a63ecb665dffcc127cf8ab
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
        self.stereo = cv2.StereoSGBM_create(minDisparity=min_disparity,
                                            numDisparities=num_disparities,
                                            blockSize=7,
                                            uniquenessRatio=uniqueness_ratio,
                                            speckleWindowSize=speckle_window_size,
                                            speckleRange=speckle_range,
                                            disp12MaxDiff=disp_max_diff,
                                            preFilterCap=pre_filter_cap,
                                            P1=p1,  # 8*3*win_size**2,
                                            P2=p2)  # 32*3*win_size**2)

        # load keypoints
        with open(os.path.join(kp_prefix, kp_path), 'rb') as f:
            self.kps = pickle.load(f)

        # load calibration matrix
        self.cal_path = cal_path

        # used only for 5-point algorithm
        _, self.cal_matrix, self.dist_matrix, _, _ = read_real_cal_matrix(cal_path, n_cam=0)

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
        self.pos = np.ones((4, 1))

    def step(self, method: str = 'five-points'):
        if self.actual_idx < self.last_frame:
            actual_frame_path = os.path.join(self.left_camera_path, rf'0000000{str(self.actual_idx).zfill(3)}.png')
            actual_frame = cv2.imread(actual_frame_path)
            actual_frame_copy = np.copy(actual_frame)

            # load the keypoints and descriptors
            actual_kp = SerializableKp.serializable2cv(self.kps[os.path.basename(actual_frame_path)]['kp'])
            actual_des = self.kps[os.path.basename(actual_frame_path)]['des']

            # match them
            kp1, kp2, matches = detect(self.pre_kp, actual_kp, self.pre_des, actual_des)

            # draw motion arrows in actual frame
            draw_motion(actual_frame, kp1, kp2, matches)

            # estimate motion
            points1, points2 = filter_kp(kp1, kp2, matches)

            if method == 'five-points':
                E, mask = cv2.findEssentialMat(points1, points2, self.cal_matrix, method=cv2.RANSAC,
                                               prob=0.999, threshold=1.0)
                _, R, t, _ = cv2.recoverPose(E, points1, points2, cameraMatrix=self.cal_matrix)
                disparity_map = None

            elif method == 'pnp':
                # read the calibration parameters
                _, k1, d1, _, _ = read_real_cal_matrix(self.cal_path, n_cam=0)
                s, k2, d2, rotation, translation = read_real_cal_matrix(self.cal_path, n_cam=1)

                # find projections and maps
                R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(k1, d1, k2, d2, s, rotation, translation)
                l_transform1, l_transform2 = cv2.initUndistortRectifyMap(k1, d1, R1, P1, s, cv2.CV_32FC1)
                r_transform1, r_transform2 = cv2.initUndistortRectifyMap(k2, d2, R2, P2, s, cv2.CV_32FC1)

                # load left and right frame
                l_frame = cv2.imread(self.pre_img_path)
                pre_right_img_path = os.path.join(self.right_camera_path,
                                                  rf'0000000{str(self.actual_idx - 1).zfill(3)}.png')
                r_frame = cv2.imread(pre_right_img_path)

                # apply maps to each frame
                # l_frame_map = cv2.remap(l_frame, l_transform1, l_transform2, None)
                # r_frame_map = cv2.remap(r_frame, r_transform1, r_transform2, None)

                # here we are not using the mapped frames
                disparity_map = self.stereo.compute(l_frame, r_frame)
                disparity_map = cv2.normalize(disparity_map, disparity_map, alpha=0, beta=255,
                                              norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                # re-project to 3D space
                # points_3d = cv2.reprojectImageTo3D(disparity_map, Q)
                # points1_3d = np.array([points_3d[points1[idx, 1], points1[idx, 0]] for idx in range(points1.shape[0])])
                # points1_3d = points1_3d[~np.isinf(points1_3d).any(axis=1)]
                # R, t = cv2.solvePnPRansac(points1_3d, points2, self.cal_matrix, self.dist_matrix)

                # just to adapt: THIS PART SHOULD BE DELETED
                E, mask = cv2.findEssentialMat(points1, points2, self.cal_matrix, method=cv2.RANSAC,
                                               prob=0.999, threshold=1.0)
                _, R, t, _ = cv2.recoverPose(E, points1, points2, cameraMatrix=self.cal_matrix)

                # cv2.imshow('Left frame', l_frame)
                # cv2.imshow('Disparity map', disparity_map)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
            else:
                raise Exception(f"Method {method} not allowed.")

            T = np.vstack([np.hstack([R, t]), [0, 0, 0, 1]])
            self.pos = T @ self.pos

            x, y = self.pos[0][0], self.pos[2][0]
            print(f"{x:.3f}, {y:.3f}")

            self.pre_kp = actual_kp
            self.pre_des = actual_des
            self.pre_img_path = actual_frame_path

            self.actual_idx += 1

            # return raw frame, motion frame, and predicted camera position
            return actual_frame_copy, actual_frame, (x, y), disparity_map
        else:
            return None, None, None, None


if __name__ == '__main__':

    date = '2011_09_26'
    id = '0005'
    video = f'{date}/{date}_drive_{id}_sync'
    keypoints = f'{date}/{date}_drive_{id}_sync/image_00/sift_keypoints.pkl'
    experimental_cal_path = f'data/{date}/calib_cam_to_cam.txt'
    v = VO(video, experimental_cal_path, keypoints)

    traj = np.zeros((600, 600, 3), dtype=np.uint8)
    raw_frame, motion_frame, pos = v.step(method='pnp')
    while pos:
        x_pos, y_pos = int(pos[0] + 300), int(pos[1] + 300)
        # print(f"Pos: {x_pos}, {y_pos}")
        cv2.circle(traj, (x_pos, traj.shape[1] - y_pos), 1, (255, 0, 0), 1)

        # show images
        cv2.imshow('Motion', motion_frame)
        cv2.imshow('Real', raw_frame)
        cv2.imshow('Trajectory', traj)
        cv2.waitKey(50)

        raw_frame, motion_frame, pos = v.step()
