import cv2


class StereoSGBMConfig:
    min_disparity = 0
    num_disparities = 16*10
    sad_window_size = 3
    uniqueness_ratio = 5
    p1 = 16*sad_window_size*sad_window_size
    p2 = 96*sad_window_size*sad_window_size
    pre_filter_cap = 63
    speckle_window_size = 0
    speckle_range = 0
    disp_max_diff = 1
    mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY


class StereoSGBMConfig2:
    pre_filter_cap = 63
    sad_window_size = 3
    p1 = sad_window_size * sad_window_size * 4
    p2 = sad_window_size * sad_window_size * 32
    min_disparity = 0
    num_disparities = 128
    uniqueness_ratio = 10
    speckle_window_size = 100
    speckle_range = 32
    disp_max_diff = 1
    full_dp = 1
    mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY


class MatcherConfig:
    ransac = {
        'iterations': 15,
        'max_iterations': 50,
        'error_threshold': 10,
        'min_consensus': 5
    }
    hough = {
        'dxbin': 60,
        'dangbin': 30,
        'umbralvotos': 10
    }
