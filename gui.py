import PySimpleGUI as sg
import cv2
import numpy as np

from odometry import VO
from extract_keypoints import SerializableKp
import argparse
import time


def main(v_engine, args):
    """
    Simple graphical interface used to visualize the results of the Visual Odometry Class.
    """

    sg.theme("DarkBlue1")

    col_1 = [
            [sg.Image(filename="", key="-IMAGE1-")],
            [sg.Image(filename="", key="-IMAGE2-")],
        ]

    col_2 = [
        [sg.Text("Trayectoria", size=(60, 1), justification="left")],
        [sg.Image(filename="", key="-IMAGE3-")]
    ]

    # Define the window layout
    layout = [
       [sg.Frame(layout=col_1, title=''), sg.Frame(layout=col_2, title='')],
       [sg.Button("Exit", size=(10, 1))]
    ]
    window = sg.Window("OpenCV Integration", layout)

    traj_w = traj_h = 600
    x_offset = traj_w / 2
    y_offset = traj_h / 2
    traj = np.zeros((traj_w, traj_h, 3), dtype=np.uint8)
    raw_frame, motion_frame, pos, disparity_map = v_engine.step(args.method_matching, args.method_track)

    if lines:
        # read initial ground truth position
        vals = list(map(float, lines[0].split(" ")))
        mat = np.array(vals).reshape((3, 4))
        _, pos_gt = mat[:, :3], mat[:, 3].reshape((3, 1))
        pos_gt_0 = np.vstack([pos_gt, 1])
    counter = 1

    trajectory_raw = []
    while pos:
        # handle window events
        event, values = window.read(timeout=20)
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        # plot predicted trajectory
        x_pos, y_pos = int(pos[0] + x_offset), int(pos[1] + y_offset)
        trajectory_raw.append((x_pos, y_pos))
        cv2.circle(traj, (x_pos, traj.shape[1] - y_pos), 1, (255, 0, 0), 1)

        if lines:
            # plot real trajectory
            x_gt_pos, y_gt_pos = int(pos_gt[0] + x_offset), int(pos_gt[2] + y_offset)
            cv2.circle(traj, (x_gt_pos, traj.shape[1] - y_gt_pos), 1, (0, 255, 0), 1)

            # get next ground truth position
            line = lines[counter]
            vals = list(map(float, line.split(" ")))
            mat = np.array(vals).reshape((3, 4))
            R, t = mat[:, :3], mat[:, 3].reshape((3, 1))
            T = np.vstack([np.hstack([R, t]), [0, 0, 0, 1]])
            pos_gt = T @ pos_gt_0

        if args.method_track == 'pnp':
            window["-IMAGE1-"].update(data=cv2.imencode(".png", motion_frame)[1].tobytes())
            window["-IMAGE2-"].update(data=cv2.imencode(".png", disparity_map)[1].tobytes())

        else:
            window["-IMAGE1-"].update(data=cv2.imencode(".png", raw_frame)[1].tobytes())
            window["-IMAGE2-"].update(data=cv2.imencode(".png", motion_frame)[1].tobytes())

        window["-IMAGE3-"].update(data=cv2.imencode(".png", traj)[1].tobytes())

        raw_frame, motion_frame, pos, disparity_map = v_engine.step(args.method_matching, args.method_track)

        counter += 1

    cv2.imwrite(f'trajectories/{date}/{date}_drive_{id}_sync/{args.method_matching}-{args.method_track}.png', traj)
    np.save(f'trajectories/{date}/{date}_drive_{id}_sync/{args.method_matching}-{args.method_track}.npy', trajectory_raw)
    window.close()


if __name__ == '__main__':
    from configs import *
    date = '2011_09_26'
    id = '0001'

    video = f'{date}/{date}_drive_{id}_sync'
    keypoints = f'{date}/{date}_drive_{id}_sync/image_00/sift_keypoints.pkl'
    experimental_cal_path = f'data/{date}/calib_cam_to_cam.txt'
    ground_truths = {
        '2011_09_30': {
            '0020': '06',
            '0027': '07'
        }
    }
    try:
        ground_truth = f'data/odometry/poses/{ground_truths[date][id]}.txt'
        with open(ground_truth, 'r') as f:
            lines = f.readlines()
    except KeyError:
        lines = None

    # argument parser
    parser = argparse.ArgumentParser(description='Model eval')
    parser.add_argument('-t', '--method_track', default="five-points", type=str,
                        help='Method using for motion estimation.')
    parser.add_argument('-m', '--method_matching', default="hough", type=str,
                        help='Method using for feature matching.')
    args = parser.parse_args()
    print(f"Motion estimated using {args.method_track}-{args.method_matching}")
    m_conf = MatcherConfig.ransac if args.method_matching == 'ransac' else MatcherConfig.hough

    v = VO(video, experimental_cal_path, keypoints, matcher_config=m_conf)
    main(v, args)
