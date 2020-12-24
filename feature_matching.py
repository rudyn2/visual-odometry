"""
La mayor parte de este código fue extraído de mi implementación de la tarea 3 del curso acerca
de Feature Matching usando Transformada de Hough y RANSAC
"""


import math
import time
import pickle
import cv2
import numpy as np
import sparse


def gen_transform(match, keypoints_query, keypoints_reference):
    # Calcular transformacion de semejanza (e,theta,tx,ty) a partir de un calce "match".
    # Se debe notar que los keypoints de OpenCV tienen orientacion en grados.
    kp_query = keypoints_query[match.queryIdx]
    kp_reference = keypoints_reference[match.trainIdx]

    e = kp_query.size / kp_reference.size
    theta = kp_query.angle - kp_reference.angle
    theta_rads = np.deg2rad(theta)
    tx = kp_query.pt[0] - e * (kp_reference.pt[0] * np.cos(theta_rads) - kp_reference.pt[1] * np.sin(theta_rads))
    ty = kp_query.pt[1] - e * (kp_reference.pt[0] * np.sin(theta_rads) + kp_reference.pt[1] * np.cos(theta_rads))

    return e, theta, tx, ty


def transform_error(e, theta, tx, ty, keypoint_query, keypoint_reference):
    theta_rads = np.deg2rad(theta)
    rotation = np.array([[np.cos(theta_rads), -np.sin(theta_rads)],
                         [np.sin(theta_rads), np.cos(theta_rads)]])
    kp_ref_pos = np.array([keypoint_reference.pt[0], keypoint_reference.pt[1]])
    kp_query_pos = np.array([keypoint_query.pt[0], keypoint_query.pt[1]])
    translation = np.array([tx, ty])

    return np.linalg.norm(e * (rotation @ kp_ref_pos) + translation - kp_query_pos)


def ransac(matches, keypoints_query, keypoints_ref):
    accepted = []
    iterations = 0
    max_iterations = 100
    error_threshold = 20
    min_matches_in_consensus_to_accept = 10

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
        print(f"At most {len(max_consensus)} matches in consensus were found. [accepted]")
        return max_consensus

    print(f"At most {len(max_consensus)} matches in consensus were found. [not accepted]")
    return accepted


def hough4d(matches, keypoints_query, keypoints_reference):
    stored = []
    # Parametros de Hough
    dxBin = 60
    dangBin = 30 * math.pi / 180
    umbralvotos = 4
    accepted = []

    # Por hacer:
    # Se debe recorrer todos los calces en "matches" y, para cada uno, calcular una
    # transformación usando genTransform(), y luego calcular los índices de la celda i,j,k,z
    # en la cual hay que hacer la votacion. Se recomienda usar un offset de 500 al acceder las celdas para evitar
    # evaluar la matriz con indices negativos
    # Los indices [i+500,j+500,k+500,z+500] se deben guardar en la lista "stored"

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

    # Por hacer:
    # Calcular la maxima cantidad de votos usando np.max(sm.data)
    max_votes = np.max(sm.data)
    most_voted_cell_idxs = sm.coords[:, np.argmax(sm.data)]
    # Si la cantidad de votos es menor que un umbral, retornar una lista vacia
    if max_votes < umbralvotos:
        return []

    # Luego, se debe recorrer nuevamente todos los calces en "matches" y calcular indices (i,j,k,z)
    # La cantidad de votos para esa celda es: sm[i+500,j+500,k+500,z+500]
    # Las correspondencias que voten por la celda mas votada se deben agregar a accepted
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


def calc_afin(matches, keypoints_query, keypoints_reference):
    # Inicializa matrices
    A = np.zeros(shape=(0, 6))
    b = np.zeros(shape=(0, 1))

    # Se construye la matriz A y b que permitirán
    # encontrar la transformación afín principal entre
    # un conjunto de matches
    for idx, match in enumerate(matches):
        x_query, y_query = keypoints_query[match.queryIdx].pt
        x_ref, y_ref = keypoints_reference[match.trainIdx].pt
        A = np.vstack([A, adapt_point(x_query, y_query)])
        b = np.vstack([b, np.array([
            [x_ref],
            [y_ref]
        ])])

    return np.linalg.inv(A.T @ A) @ (A.T @ b)


def draw_proj_afin(transf, input1, input2):
    # Dibuja un romboide que representa el rectangulo
    # de la imagen "input2" proyectada en la imagen "input1"

    # Proyecta esquinas de la imagen
    upper_left_t = adapt_point(0, 0) @ transf
    bottom_left_t = adapt_point(input1.shape[1] - 1, 0) @ transf
    upper_right_t = adapt_point(0, input1.shape[0] - 1) @ transf
    bottom_right = adapt_point(input1.shape[1] - 1, input1.shape[0] - 1) @ transf

    # Dibuja un polígono a partir de las esquinas proyectadas
    pts = np.array([upper_left_t, bottom_left_t, bottom_right, upper_right_t], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(input2, [pts], True, (255, 0, 0))

    return input2


def filter_matches(matches, kp1, kp2):
    # Apply ratio test
    points1 = []
    points2 = []
    good = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:  # 0.75
            good.append(m)
            points1.append(kp1[m.queryIdx].pt)
            points2.append(kp2[m.trainIdx].pt)
    return np.array(points1), np.array(points2), good


def detect(kp_ref: list, kp_query: list, des_ref, des_query, method: str = 'hough', save: bool = False):

    # create matches
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_query, des_ref, k=2)
    points1, points2, good = filter_matches(matches, kp_query, kp_ref)

    if method == 'ransac':
        accepted = ransac(good, kp_query, kp_ref)
    elif method == 'hough':
        accepted = hough4d(good, kp_query, kp_ref)
    else:
        raise ValueError(f"Method {method} not known. Available: ransac, hough.")

    return kp_ref, kp_query, accepted


def draw_motion(img_to_plot, kp_ref, kp_query, matches):
    for match in matches:
        pt2 = tuple(map(int, kp_ref[match.trainIdx].pt))
        pt1 = tuple(map(int, kp_query[match.queryIdx].pt))
        cv2.arrowedLine(img_to_plot, pt1, pt2, (0, 255, 255), thickness=1, line_type=cv2.LINE_AA)
        # cv2.drawKeypoints(img_to_plot, kp_query, img_to_plot, color=(255, 0, 0), flags=cv2.DrawMatchesFlags_DEFAULT)
    return img_to_plot


if __name__ == '__main__':
    from extract_keypoints import SerializableKp
    import os
    import glob

    # video = '2011_09_26/2011_09_26_drive_0001_sync/image_00/'
    video = '2011_09_30/2011_09_30_drive_0016_sync/image_00'
    with open(os.path.join('keypoints', video, 'sift_keypoints.pkl'), 'rb') as f:
        kps = pickle.load(f)

    img_path = os.path.join('data', video, 'data')
    pre_kp = SerializableKp.serializable2cv(kps['0000000000.png']['kp'])
    pre_des = kps['0000000000.png']['des']

    for idx in range(1, len(glob.glob(os.path.join(img_path, '*.png')))):
        actual_frame_path = os.path.join(img_path, rf'0000000{str(idx).zfill(3)}.png')
        actual_frame = cv2.imread(actual_frame_path)

        actual_kp = SerializableKp.serializable2cv(kps[os.path.basename(actual_frame_path)]['kp'])
        actual_des = kps[os.path.basename(actual_frame_path)]['des']

        kp1, kp2, matches = detect(pre_kp, actual_kp, pre_des, actual_des)

        draw_motion(actual_frame, kp1, kp2, matches)
        cv2.imshow('Motion', actual_frame)
        cv2.waitKey(50)

        pre_kp = actual_kp
        pre_des = actual_des

    cv2.destroyAllWindows()
