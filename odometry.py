from feature_matching import detect
import cv2
import pickle
import numpy as np
import os
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
    def __init__(self, video: str, cal_path: str, kp_path: str, video_prefix: str = 'data', kp_prefix: str = 'keypoints'):

        # load keypoints
        with open(os.path.join(kp_prefix, kp_path), 'rb') as f:
            self.kps = pickle.load(f)

        # load calibration matrix
        self.camera_matrix, _ = read_real_cal_matrix(cal_path)

        # frames directory
        self.img_path = os.path.join(video_prefix, video, 'data')

        # init first image and its keypoints, descriptors
        self.pre_img_path = os.path.join(self.img_path, '0000000000.png')
        self.pre_kp = SerializableKp.serializable2cv(self.kps['0000000000.png']['kp'])
        self.pre_des = self.kps['0000000000.png']['des']
        self.actual_idx = 1
        self.last_frame = len(os.listdir(self.img_path))

        # position tracking
        self.pos = np.ones((4, 1))

    def step(self, method: str = 'five-points'):
        if self.actual_idx < self.last_frame:
            actual_frame_path = os.path.join(self.img_path, rf'0000000{str(self.actual_idx).zfill(3)}.png')
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
                E, mask = cv2.findEssentialMat(points1, points2, self.camera_matrix, method=cv2.RANSAC,
                                               prob=0.999, threshold=1.0)
                _, R, t, _ = cv2.recoverPose(E, points1, points2, cameraMatrix=self.camera_matrix)
                T = np.vstack([np.hstack([R, t]), [0, 0, 0, 1]])
            self.pos = T @ self.pos
            x, y = self.pos[0][0], self.pos[2][0]
            print(f"{x:.3f}, {y:.3f}")

            self.pre_kp = actual_kp
            self.pre_des = actual_des
            self.pre_img_path = actual_frame_path

            self.actual_idx += 1

            # return raw frame, motion frame, and predicted camera position
            return actual_frame_copy, actual_frame, (x, y)
        else:
            return None, None, None


if __name__ == '__main__':

    date = '2011_09_26'
    id = '0005'
    video = f'{date}/{date}_drive_{id}_sync/image_00/'
    keypoints = f'{date}/{date}_drive_{id}_sync/image_00/sift_keypoints.pkl'
    experimental_cal_path = f'data/{date}/calib_cam_to_cam.txt'
    v = VO(video, experimental_cal_path, keypoints)

    traj = np.zeros((600, 600, 3), dtype=np.uint8)
    raw_frame, motion_frame, pos = v.step()
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

