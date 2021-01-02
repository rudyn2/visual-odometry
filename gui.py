import PySimpleGUI as sg
import cv2
import numpy as np

from odometry import VO
from extract_keypoints import SerializableKp
import argparse


def main(v_engine, args):
    sg.theme("DarkBlue1")

    col_1 = [
            [sg.Text("Video + motion", size=(60, 1), justification="left")],
            [sg.Image(filename="", key="-IMAGE1-")],
            [sg.Text("Depth map", size=(60, 1), justification="left")],
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
    traj = np.zeros((traj_w, traj_h, 3), dtype=np.uint8)
    raw_frame, motion_frame, pos, disparity_map = v_engine.step(method_match, args.method_track)
    while pos:
        # handle window events
        event, values = window.read(timeout=20)
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        x_pos, y_pos = int(pos[0] + traj_w / 2), int(pos[1] + traj_h / 2)
        cv2.circle(traj, (x_pos, traj.shape[1] - y_pos), 1, (255, 0, 0), 1)

        if args.method_track == 'pnp':
            window["-IMAGE1-"].update(data=cv2.imencode(".png", motion_frame)[1].tobytes())
            window["-IMAGE2-"].update(data=cv2.imencode(".png", disparity_map)[1].tobytes())

        else:
            window["-IMAGE1-"].update(data=cv2.imencode(".png", raw_frame)[1].tobytes())
            window["-IMAGE2-"].update(data=cv2.imencode(".png", motion_frame)[1].tobytes())

        window["-IMAGE3-"].update(data=cv2.imencode(".png", traj)[1].tobytes())

        raw_frame, motion_frame, pos, disparity_map = v_engine.step(method_match, args.method_track)
    window.close()


if __name__ == '__main__':
    from configs import *
    date = '2011_09_26'
    id = '0005'
    video = f'{date}/{date}_drive_{id}_sync'
    keypoints = f'{date}/{date}_drive_{id}_sync/image_00/sift_keypoints.pkl'
    experimental_cal_path = f'data/{date}/calib_cam_to_cam.txt'

    # argument parser
    parser = argparse.ArgumentParser(description='Model eval')
    parser.add_argument('-t', '--method_track', default="five-points", type=str,
                        help='Method using for motion estimation.')
    args = parser.parse_args()
    print(f"Motion estimated using {args.method_track}")

    method_match = 'hough'
    method_match_config = MatcherConfig.hough

    v = VO(video, experimental_cal_path, keypoints, matcher_config=method_match_config)
    main(v, args)
