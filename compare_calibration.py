import pickle
import numpy as np


def process_line(s: str, shape: tuple):
    values = [float(c.strip()) for c in s.split(" ")[1:]]
    values = np.array(values)
    values = np.reshape(values, shape)
    return values


def read_real_cal_matrix(cal_path: str):
    # read dataset calibration parameters
    with open(cal_path, 'r') as f:
        lines = f.readlines()
    k_real = process_line(lines[3], (3, 3))
    d_real = process_line(lines[4], (1, 5))
    return k_real, d_real


if __name__ == '__main__':
    calculated_cal_path = 'calibration_results.pkl'
    experimental_cal_path = 'data/2011_09_26/calib_cam_to_cam.txt'

    # read calculated calibration parameters
    with open(calculated_cal_path, 'rb') as f:
        cal_data = pickle.load(f)
    k_cal = cal_data['K']
    d_cal = cal_data['D']
    k_real, d_real = read_real_cal_matrix(experimental_cal_path)


    # compute RMSE
    k_rmse = np.sqrt(np.sum((k_cal.flatten() - k_real.flatten()) ** 2) / k_cal.flatten().shape[0])
    d_rmse = np.sqrt(np.sum((d_cal.flatten() - d_real.flatten())**2) / d_cal.flatten().shape[0])
    print(f"K_RMSE: {k_rmse:.3f}\nD_RMSE: {d_rmse:.3f}")







