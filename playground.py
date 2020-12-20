from datetime import datetime
import pickle
import cv2 as cv

if __name__ == '__main__':

    with open('calibration_results.pkl', 'rb') as input:
        cal_data = pickle.load(input)
    img = cv.imread('2011_09_26/2011_09_26/2011_09_26_drive_0119_extract/image_00/data/0000000000.png')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    objpoints = cal_data['obj']
    imgpoints = cal_data['img']
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)