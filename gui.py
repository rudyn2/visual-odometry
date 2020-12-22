import PySimpleGUI as sg
import cv2
import numpy as np

from odometry import VO
from extract_keypoints import SerializableKp


def main(v_engine):
    sg.theme("DarkBlue1")

    col_1 = [
            [sg.Text("Video original", size=(60, 1), justification="left")],
            [sg.Image(filename="", key="-IMAGE1-")],
            [sg.Text("Visualización Keypoints matching", size=(60, 1), justification="left")],
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

    traj = np.zeros((600, 600, 3), dtype=np.uint8)
    raw_frame, motion_frame, pos = v_engine.step()
    while pos:
        # handle window events
        event, values = window.read(timeout=20)
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        x_pos, y_pos = int(pos[0] + 300), int(pos[1] + 300)
        cv2.circle(traj, (x_pos, traj.shape[1] - y_pos), 1, (255, 0, 0), 1)

        window["-IMAGE1-"].update(data=cv2.imencode(".png", raw_frame)[1].tobytes())
        window["-IMAGE2-"].update(data=cv2.imencode(".png", motion_frame)[1].tobytes())
        window["-IMAGE3-"].update(data=cv2.imencode(".png", traj)[1].tobytes())

        raw_frame, motion_frame, pos = v_engine.step()
    window.close()


if __name__ == '__main__':
    date = '2011_09_26'
    id = '0005'
    video = f'{date}/{date}_drive_{id}_sync/image_00/'
    keypoints = f'{date}/{date}_drive_{id}_sync/image_00/sift_keypoints.pkl'
    experimental_cal_path = f'data/{date}/calib_cam_to_cam.txt'
    v = VO(video, experimental_cal_path, keypoints)
    main(v)
