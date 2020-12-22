from feature_matching import detect
import cv2
import pickle
import numpy as np
import os


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


if __name__ == '__main__':
    # read calculated calibration parameters
    calculated_cal_path = 'calibration_results.pkl'
    experimental_cal_path = 'data/2011_09_26/calib_cam_to_cam.txt'
    with open(calculated_cal_path, 'rb') as f:
        cal_data = pickle.load(f)
    camera_matrix = cal_data['K']

    img_path = 'data/2011_09_26/2011_09_26_drive_0005_sync/image_00/data'
    pre_img_path = os.path.join(img_path, rf'0000000000.png')
    pos = np.zeros((3, 1))
    traj = np.zeros((600, 600, 3), dtype=np.uint8)

    for idx in range(1, 153):
        actual_frame_path = os.path.join(img_path, rf'0000000{str(idx).zfill(3)}.png')
        img1, kp1, img2, kp2, matches = detect(pre_img_path, actual_frame_path, method='hough')
        points1, points2 = filter_kp(kp1, kp2, matches)

        # Estimate motion
        E, mask = cv2.findEssentialMat(points1, points2, camera_matrix,  method=cv2.RANSAC,
                                       prob=0.999, threshold=1.0)
        _, R, t, _ = cv2.recoverPose(E, points1, points2, cameraMatrix=camera_matrix)
        # T = np.vstack([np.hstack([R, t]), [0, 0, 0, 1]])
        pos = R @ (pos - t)
        x, y, z = pos[0], pos[1], pos[2]

        draw_x, draw_y = int(x) + 300, int(z) + 400
        cv2.circle(traj, (draw_x, draw_y), 1, (255, 0, 0), 1)
        cv2.rectangle(traj, (10, 20), (600, 60), (0, 0, 0), -1)
        text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z)
        cv2.putText(traj, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

        cv2.imshow('Real', cv2.imread(actual_frame_path))
        cv2.imshow('Trajectory', traj)
        cv2.waitKey(1)

        pre_img_path = actual_frame_path
