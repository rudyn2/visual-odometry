import glob
import os
import pickle

import cv2 as cv


class SerializableKp:

    def __init__(self, angle: float, class_id: int, octave: int, pt: tuple, response: float, size: float):
        self.angle = angle
        self.class_id = class_id
        self.octave = octave
        self.pt = pt
        self.response = response
        self.size = size

    @classmethod
    def cv2serializable(cls, kp: cv.KeyPoint):
        return cls(kp.angle, kp.class_id, kp.octave, kp.pt, kp.response, kp.size)

    @classmethod
    def serializable2cv(cls, kp):
        cv_kps = []
        for k in kp:
            cv_kps.append(cv.KeyPoint(x=k.pt[0], y=k.pt[1], _angle=k.angle, _class_id=k.class_id, _octave=k.octave, _response=k.response,
                                      _size=k.size))
        return cv_kps


def extract_keypoints(img_dir: str, output_path: str, method: str):
    sift = cv.SIFT_create()
    orb = cv.ORB_create()
    kps = {}
    for idx, image in enumerate(glob.glob(os.path.join(img_dir, "*.png"))):
        image_ = cv.imread(image, 0)
        if method == 'sift':
            image_kp, image_descriptors = sift.detectAndCompute(image_, None)
        elif method == 'orb':
            image_kp, image_descriptors = orb.detectAndCompute(image_, None)
        else:
            raise Exception("You must select a sift or orb detector.")

        image_kp_serial = [SerializableKp.cv2serializable(kp) for kp in image_kp]

        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        kps[os.path.basename(image)] = {
            'kp': image_kp_serial,
            'des': image_descriptors
        }

    with open(output_path, 'wb') as output:
        pickle.dump(kps, output, pickle.HIGHEST_PROTOCOL)
    print(f"Keypoints saved successfully at {output_path}")


if __name__ == '__main__':
    img_dir = 'data/2011_09_26/2011_09_26_drive_0001_sync/image_00/data/'
    detector = 'sift'
    output_dir = f'keypoints/2011_09_26/2011_09_26_drive_0001_sync/image_00/{detector}_keypoints.pkl'
    extract_keypoints(img_dir, output_dir, detector)
