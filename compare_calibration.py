import pickle
import numpy as np


def process_line(s: str, shape: tuple):
    values = [float(c.strip()) for c in s.split(" ")[1:]]
    values = np.array(values)
    values = np.reshape(values, shape)
    return values


def read_real_cal_matrix(cal_path: str, n_cam: int = 0):
    # read dataset calibration parameters
    offset_headers = 2
    with open(cal_path, 'r') as f:
        lines = f.readlines()
    s = process_line(lines[offset_headers + n_cam * 8], (2, 1))
    s = int(s[0][0]), int(s[1][0])
    k = process_line(lines[offset_headers + n_cam*8 + 1], (3, 3))
    d = process_line(lines[offset_headers + n_cam*8 + 2], (1, 5))
    R = process_line(lines[offset_headers + n_cam*8 + 3], (3, 3))
    T = process_line(lines[offset_headers + n_cam*8 + 4], (3, 1))
    return s, k, d, R, T


if __name__ == '__main__':
    calculated_cal_path = 'calibration_results.pkl'
    experimental_cal_path = 'data/2011_09_26/calib_cam_to_cam.txt'

    # read calculated calibration parameters
    with open(calculated_cal_path, 'rb') as f:
        cal_data = pickle.load(f)
    k_cal = cal_data['K']
    d_cal = cal_data['D']
    k_real, d_real, _, _, _ = read_real_cal_matrix(experimental_cal_path)


    # compute RMSE
    k_rmse = np.sqrt(np.sum((k_cal.flatten() - k_real.flatten()) ** 2) / k_cal.flatten().shape[0])
    d_rmse = np.sqrt(np.sum((d_cal.flatten() - d_real.flatten())**2) / d_cal.flatten().shape[0])
    print(f"K_RMSE: {k_rmse:.3f}\nD_RMSE: {d_rmse:.3f}")







