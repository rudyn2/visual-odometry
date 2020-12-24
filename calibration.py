import glob
import json
import os
import pickle

import cv2 as cv
import numpy as np
from tqdm import tqdm


def calibrate_chessboards(chessboards: list, visualize: bool = False):
    finder_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
    total_obj_points, total_image_points = [], []
    chessboards_points = {}
    total_points = 0
    chess_iter = tqdm(enumerate(chessboards), "Chessboard ")
    for idx, chessboard in chess_iter:
        chessboards_points = 0

        for i_grid in range(5, 13):
            for j_grid in range(5, 13):
                if j_grid >= i_grid:
                    continue

                # Try to find the chess board corners
                ret, chess_corners = cv.findChessboardCorners(chessboard, (i_grid, j_grid), None, finder_criteria)
                # If found, add object points, image points (after refining them)
                if ret:
                    objp = np.zeros((i_grid * j_grid, 3), np.float32)
                    objp[:, :2] = np.mgrid[0:i_grid, 0:j_grid].T.reshape(-1, 2)
                    chess_corners = cv.cornerSubPix(chessboard, chess_corners, (5, 5), (-1, -1), criteria)

                    if visualize:
                        color_chessboard = cv.cvtColor(np.copy(chessboard), cv.COLOR_GRAY2BGR)
                        cv.drawChessboardCorners(color_chessboard, (i_grid, j_grid), chess_corners, True)
                        cv.imshow(f'Chess: {idx} {i_grid}x{j_grid}', color_chessboard)
                        cv.waitKey(5000)

                    total_obj_points.append(objp)
                    total_image_points.append(chess_corners)
                    chessboards_points += i_grid * j_grid
                    chess_iter.set_description(f"Chessboard {idx}: {chessboards_points}pts")

        total_points += chessboards_points

    return total_obj_points, total_image_points, total_points


def generate_calibration_images(image_path: str, annotation_path: str, crop: bool = False) -> list:
    img = cv.imread(image_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    with open(annotation_path, 'r') as f:
        annotations = json.load(f)

    result_images = []
    for annotation in annotations.values():
        x, y, w, h = annotation.values()
        if crop:
            img_annotated = img[y:y + h, x:x + w]
        else:
            img_annotated = np.zeros_like(img)
            img_annotated[y:y + h, x:x + w] = img[y:y + h, x:x + w]
        result_images.append(img_annotated)

    return result_images


if __name__ == '__main__':

    img_dir = 'data/2011_09_26/2011_09_26_drive_0119_extract/image_00/data'
    calibration_path = 'data/2011_09_26/2011_09_26_image_00_chessboards.json'
    output_path = 'calibration_results.pkl'

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    best_points = 0
    best_objpoints, best_imgpoints, best_image = None, None, None
    total_objpoints = []
    total_imgpoints = []
    visualize = False
    use_all_images = False

    for idx, image in enumerate(glob.glob(os.path.join(img_dir, "*.png"))):
        if idx > 0:
            break
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
        print(f"Calibrating using all chessboard in image with {best_points} corners detected.")
        ret, mtx, dist, rvecs, tvecs  = cv.calibrateCamera(best_objpoints, best_imgpoints, best_image.shape[::-1], None,
                                                          None)
        print(f"RMS Reprojection error: {ret:.4f}")

    if ret:
        print(f"Saving results on {output_path}")
        with open(output_path, 'wb') as output:
            results = {
                'K': mtx,
                'D': dist
            }
            pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)
        print("The calibration has ended successfully!")
