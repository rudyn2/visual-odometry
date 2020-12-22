import glob
import json
import os
import pickle

import cv2 as cv
import numpy as np
from tqdm import tqdm


def calibrate_chessboards(chessboards: list, visualize: bool = False):
    total_obj_points, total_image_points = [], []
    total_points = 0
    for idx, chessboard in tqdm(enumerate(chessboards), f"Chessboard "):
        chessboard_points = 0
        for i_grid in range(5, 13):
            for j_grid in range(5, 13):
                # Try to find the chess board corners
                ret, chess_corners = cv.findChessboardCorners(chessboard, (i_grid, j_grid), None)
                # If found, add object points, image points (after refining them)
                if ret:
                    objp = np.zeros((i_grid * j_grid, 3), np.float32)
                    objp[:, :2] = np.mgrid[0:i_grid, 0:j_grid].T.reshape(-1, 2)

                    if visualize:
                        corners_refined = cv.cornerSubPix(chessboard, chess_corners, (11, 11), (-1, -1), criteria)
                        cv.drawChessboardCorners(chessboard, (i_grid, j_grid), corners_refined, True)
                        cv.imshow('img', chessboard)
                        cv.waitKey(500)

                    total_obj_points.append(objp)
                    total_image_points.append(chess_corners)
                    chessboard_points += (i_grid * j_grid)

        total_points += chessboard_points

    return total_obj_points, total_image_points, total_points


def generate_calibration_images(image_path: str, annotation_path: str) -> list:
    img = cv.imread(image_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    with open(annotation_path, 'r') as f:
        annotations = json.load(f)

    result_images = []
    for annotation in annotations.values():
        x, y, w, h = annotation.values()
        img_annotated = np.zeros_like(img)
        img_annotated[y:y + h, x:x + w] = img[y:y + h, x:x + w]
        # img_annotated = img[y:y+h, x:x+w]
        result_images.append(img_annotated)

    return result_images


if __name__ == '__main__':

    img_dir = 'data/2011_09_26/2011_09_26_drive_0119_extract/image_00/data'
    calibration_path = 'data/2011_09_26/2011_09_26_image_00_chessboards2.json'
    output_path = 'calibration_results.pkl'
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    best_points = 0
    best_objpoints, best_imgpoints, best_image = None, None, None
    total_objpoints = []
    total_imgpoints = []
    visualize = False
    use_all_images = True

    for idx, image in enumerate(glob.glob(os.path.join(img_dir, "*.png"))):
        print(f"Analyzing img: {image}")
        # generate 1 image per chessboard
        images = generate_calibration_images(image, calibration_path)
        # find points in each chessboard
        objpoints, imgpoints, points = calibrate_chessboards(images, visualize)

        total_objpoints.extend(objpoints)
        total_imgpoints.extend(imgpoints)

        if points > best_points:
            best_points = points
            best_objpoints = objpoints
            best_imgpoints = imgpoints
            best_image = image

        print(f"Best points: {points}\n")

    best_image = cv.imread(best_image)
    best_image = cv.cvtColor(best_image, cv.COLOR_BGR2GRAY)

    if use_all_images:
        print("Calibrating using all chessboard in all images")
        # here we assume that all images are almost equal so we can use the shape of an arbitrary chosen image
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(total_objpoints, total_imgpoints, best_image.shape[::-1],
                                                          None, None)
    else:
        print("Calibrating using all chessboard in image with most corners")
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(best_objpoints, best_imgpoints, best_image.shape[::-1], None,
                                                          None)

    if ret:
        print(f"Saving results on {output_path}")
        with open(output_path, 'wb') as output:
            results = {
                'K': mtx,
                'D': dist
            }
            pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)
        print("The calibration has ended successfully!")
