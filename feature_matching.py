"""
La mayor parte de este código fue extraído de mi implementación de la tarea 3 del curso acerca
de Feature Matching usando Transformada de Hough y RANSAC
"""


import math
import pickle
import cv2
import numpy as np
import sparse
import time


def gen_transform(match, keypoints_query, keypoints_reference):
    """
    Encuentra los parámetros de una transformación de semejanza entre un par de puntos de un calce específico.
    """
    kp_query = keypoints_query[match.queryIdx]
    kp_reference = keypoints_reference[match.trainIdx]

    e = kp_query.size / kp_reference.size
    theta = kp_query.angle - kp_reference.angle
    theta_rads = np.deg2rad(theta)
    tx = kp_query.pt[0] - e * (kp_reference.pt[0] * np.cos(theta_rads) - kp_reference.pt[1] * np.sin(theta_rads))
    ty = kp_query.pt[1] - e * (kp_reference.pt[0] * np.sin(theta_rads) + kp_reference.pt[1] * np.cos(theta_rads))

    return e, theta, tx, ty


def transform_error(e, theta, tx, ty, keypoint_query, keypoint_reference):
    """
    Encuentra el error de proyección de un punto dada una transformación dada.
    """
    theta_rads = np.deg2rad(theta)
    rotation = np.array([[np.cos(theta_rads), -np.sin(theta_rads)],
                         [np.sin(theta_rads), np.cos(theta_rads)]])
    kp_ref_pos = np.array([keypoint_reference.pt[0], keypoint_reference.pt[1]])
    kp_query_pos = np.array([keypoint_query.pt[0], keypoint_query.pt[1]])
    translation = np.array([tx, ty])

    return np.linalg.norm(e * (rotation @ kp_ref_pos) + translation - kp_query_pos)


def ransac(matches, keypoints_query, keypoints_ref, **kwargs):
    """
    RANSAC Feature Matching Algorithm.
    """
    accepted = []

    iterations = 0
    max_iterations = kwargs['max_iterations']
    error_threshold = kwargs['error_threshold']
    min_matches_in_consensus_to_accept = kwargs['min_consensus']

    candidates = []
    while iterations < max_iterations:
        # selection
        match_idx = np.random.choice(range(len(matches)))
        match_selected = matches[match_idx]

        # model generation
        maybe_model = gen_transform(match_selected, keypoints_query, keypoints_ref)

        # consensus evaluation
        matches_in_consensus = [match_selected]
        for match in matches:
            if match == match_selected:
                continue

            kp_test_query = keypoints_query[match.queryIdx]
            kp_test_ref = keypoints_ref[match.trainIdx]

            if transform_error(*maybe_model, kp_test_query, kp_test_ref) < error_threshold:
                matches_in_consensus.append(match)

        candidates.append(matches_in_consensus)
        iterations += 1

    max_consensus = max(candidates, key=len)
    if len(max_consensus) > min_matches_in_consensus_to_accept:
        return max_consensus
    return accepted


def hough4d(matches, keypoints_query, keypoints_reference, **kwargs):
    """
    Hough Feature Matching Algorithm.
    """
    stored = []
    # Parametros de Hough
    dxBin = kwargs['dxbin']
    dangBin = kwargs['dangbin'] * math.pi / 180
    umbralvotos = kwargs['umbralvotos']
    accepted = []

    for match in matches:
        e, theta, tx, ty = gen_transform(match, keypoints_query, keypoints_reference)
        i = np.floor(tx / dxBin + 0.5) + 500
        j = np.floor(ty / dxBin + 0.5) + 500
        k = np.floor(np.deg2rad(theta) / dangBin + 0.5) + 500
        z = np.floor(np.log(e) / np.log(2) + 0.5) + 500
        if i < 0 or j < 0 or k < 0 or z < 0:
            continue
        stored.append([i, j, k, z])

    # Este codigo permite crear una matriz sparse para almacenar los votos
    if len(stored) == 0:
        return []
    coords = np.transpose(np.array(stored))
    data = np.ones(len(stored))
    sm = sparse.COO(coords, data, shape=((2000,) * 4))

    # Calcular la maxima cantidad de votos usando np.max(sm.data)
    max_votes = np.max(sm.data)
    most_voted_cell_idxs = sm.coords[:, np.argmax(sm.data)]

    # Si la cantidad de votos es menor que un umbral, retornar una lista vacia
    if max_votes < umbralvotos:
        return []

    for idx, match_cell in enumerate(stored):
        it_voted = [match_cell[i] == most_voted_cell_idxs[i] for i in range(len(match_cell))]
        if all(it_voted):
            accepted.append(matches[idx])

    return accepted


def adapt_point(x, y):
    return np.array([
        [x, y, 0, 0, 1, 0],
        [0, 0, x, y, 0, 1]
    ])


def filter_matches(matches, kp1, kp2):
    """
    Aplica el test de razón de Lowe a un conjunto de calces.
    """
    # Apply ratio test
    points1 = []
    points2 = []
    good = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
            points1.append(kp1[m.queryIdx].pt)
            points2.append(kp2[m.trainIdx].pt)
    return np.array(points1), np.array(points2), good


def detect(kp_ref: list, kp_query: list, des_ref, des_query, method: str = 'hough', base_matcher: str = 'brute-force', save: bool = False, **kwargs):
    """
    Ejecuta el pipeline de feature matching completo. Desde la generacion de calces por fuerza bruta hasta el filtrado
    mediante Hough o RANSAC.
    """
    if base_matcher == 'brute-force':
        matcher = cv2.BFMatcher()
    elif base_matcher == 'flann':
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=100)  # or pass empty dictionary
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        raise Exception(f"Base matcher {base_matcher} not found.")

    matches = matcher.knnMatch(des_query, des_ref, k=2)
    points1, points2, good = filter_matches(matches, kp_query, kp_ref)

    start = time.time()
    if method == 'ransac':
        accepted = ransac(good, kp_query, kp_ref, **kwargs)
    elif method == 'hough':
        accepted = hough4d(good, kp_query, kp_ref, **kwargs)
    else:
        accepted = good
    print(f"{method}: {(time.time()-start):.3f} seconds.")

    return kp_ref, kp_query, accepted


def draw_motion(img_to_plot, kp_ref, kp_query, matches, color=(0, 255, 255)):
    """
    Dibuja una flecha entre los puntos de cada calce.
    """
    for match in matches:
        pt2 = tuple(map(int, kp_ref[match.trainIdx].pt))
        pt1 = tuple(map(int, kp_query[match.queryIdx].pt))
        cv2.arrowedLine(img_to_plot, pt1, pt2, color, thickness=1, line_type=cv2.LINE_AA)
    return img_to_plot


def draw_keypoints(img_to_plot, keypoints):
    """
    Dibuja los keypoints en una imagen.
    """
    cv2.drawKeypoints(img_to_plot, keypoints, img_to_plot, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)


if __name__ == '__main__':
    from extract_keypoints import SerializableKp
    import os
    import glob
    from configs import MatcherConfig

    video = '2011_09_26/2011_09_26_drive_0005_sync/image_00/'
    # video = '2011_09_30/2011_09_30_drive_0016_sync/image_00'
    with open(os.path.join('keypoints', video, 'orb_keypoints.pkl'), 'rb') as f:
        kps = pickle.load(f)

    img_path = os.path.join('data', video, 'data')
    pre_kp = SerializableKp.serializable2cv(kps['0000000000.png']['kp'])
    pre_des = kps['0000000000.png']['des']

    for idx in range(1, len(glob.glob(os.path.join(img_path, '*.png')))):
        actual_frame_path = os.path.join(img_path, rf'0000000{str(idx).zfill(3)}.png')
        actual_frame = cv2.imread(actual_frame_path)

        actual_kp = SerializableKp.serializable2cv(kps[os.path.basename(actual_frame_path)]['kp'])
        actual_des = kps[os.path.basename(actual_frame_path)]['des']

        kp1, kp2, matches = detect(pre_kp, actual_kp, pre_des, actual_des, method='hough', **MatcherConfig.hough)

        draw_motion(actual_frame, kp1, kp2, matches)
        cv2.imshow('Motion', actual_frame)
        cv2.waitKey(10000)

        pre_kp = actual_kp
        pre_des = actual_des

    cv2.destroyAllWindows()
