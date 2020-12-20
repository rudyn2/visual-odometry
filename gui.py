import PySimpleGUI as sg
import cv2
import numpy as np
import os


def main():
    folder_path = r'2011_09_26\2011_09_26\2011_09_26_drive_0001_sync\image_03/data'
    frames = [cv2.imread(os.path.join(folder_path, rf'0000000{str(i).zfill(3)}.png')) for i in range(108)]
    sg.theme("LightGreen")

    # Define the window layout
    layout = [
        [sg.Text("OpenCV Demo", size=(60, 1), justification="center")],
        [sg.Image(filename="", key="-IMAGE-")],
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
        frame = frames[idx % len(frames)]
        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        window["-IMAGE-"].update(data=imgbytes)
        idx += 1

    window.close()

main()