import PySimpleGUI as sg
import cv2
import numpy as np
import os
import glob
from feature_matching import detect, draw_motion
from tqdm import tqdm


def main(input_dir):
    pre = None
    motion_frames = []
    for idx, image in tqdm(enumerate(glob.glob(os.path.join(input_dir, "*.png"))), "Processing"):
        if idx > 0:
            img1, kp1, img2, kp2, matches = detect(pre, image, method='hough')
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
            motion_frame = draw_motion(img2, kp1, kp2, matches)
            motion_frames.append(motion_frame)
        else:
            motion_frames.append(cv2.imread(image))
        pre = image
    sg.theme("LightGreen")

    # Define the window layout
    layout = [
        [sg.Text("Video original", size=(60, 1), justification="center")],
        [sg.Image(filename="", key="-IMAGE1-")],
        [sg.Text("Visualización Keypoints matching", size=(60, 1), justification="center")],
        [sg.Image(filename="", key="-IMAGE2-")],
        # [sg.Radio("None", "Radio", True, size=(10, 1))],
        # [
        #     sg.Radio("threshold", "Radio", size=(10, 1), key="-THRESH-"),
        #     sg.Slider(
        #         (0, 255),
        #         128,
        #         1,
        #         orientation="h",
        #         size=(40, 15),
        #         key="-THRESH SLIDER-",
        #     ),
        # ],
        # [
        #     sg.Radio("canny", "Radio", size=(10, 1), key="-CANNY-"),
        #     sg.Slider(
        #         (0, 255),
        #         128,
        #         1,
        #         orientation="h",
        #         size=(20, 15),
        #         key="-CANNY SLIDER A-",
        #     ),
        #     sg.Slider(
        #         (0, 255),
        #         128,
        #         1,
        #         orientation="h",
        #         size=(20, 15),
        #         key="-CANNY SLIDER B-",
        #     ),
        # ],
        # [
        #     sg.Radio("blur", "Radio", size=(10, 1), key="-BLUR-"),
        #     sg.Slider(
        #         (1, 11),
        #         1,
        #         1,
        #         orientation="h",
        #         size=(40, 15),
        #         key="-BLUR SLIDER-",
        #     ),
        # ],
        # [
        #     sg.Radio("hue", "Radio", size=(10, 1), key="-HUE-"),
        #     sg.Slider(
        #         (0, 225),
        #         0,
        #         1,
        #         orientation="h",
        #         size=(40, 15),
        #         key="-HUE SLIDER-",
        #     ),
        # ],
        # [
        #     sg.Radio("enhance", "Radio", size=(10, 1), key="-ENHANCE-"),
        #     sg.Slider(
        #         (1, 255),
        #         128,
        #         1,
        #         orientation="h",
        #         size=(40, 15),
        #         key="-ENHANCE SLIDER-",
        #     ),
        # ],
        [sg.Button("Exit", size=(10, 1))],
    ]

    # Create the window and show it without the plot
    window = sg.Window("OpenCV Integration", layout, location=(800, 400))
    idx = 0
    while True:
        event, values = window.read(timeout=20)
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        motion_frame = motion_frames[idx % len(motion_frames)]
        # window["-IMAGE1-"].update(data=cv2.imencode(".png", frame)[1].tobytes())
        window["-IMAGE1-"].update(data=cv2.imencode(".png", motion_frame)[1].tobytes())
        idx += 1

    window.close()


if __name__ == '__main__':
    video_dir = 'data/2011_09_26/2011_09_26_drive_0001_sync/image_00/data'
    main(video_dir)