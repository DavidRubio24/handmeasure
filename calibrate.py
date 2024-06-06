"""
This script is independent of the rest of the project.
It is used to calibrate the images from the camera.

It is used to correct the perspective and eye fish distortions of the images.

It uses a predefined intrinsic matrix and extrinsic parameters that
have been empirically obtained by calibrating the camera with OpenCV.
It can load another calibration from a JSON.
"""

import os
import argparse

import cv2
import numpy as np

FOCAL_LENGTH = 370000     # Empirical: based on OpenCV calibration.
SHAPE = (3120, 4208, 3)   # Max camera resolution
INTRINSIC_MATRIX = np.array([[FOCAL_LENGTH, 0, SHAPE[1] // 2 - 800],  # The optical center has empirically been found off-centered
                             [0, FOCAL_LENGTH, SHAPE[0] // 2 - 500],  # by (500, 800) pixels from the image center.
                             [0,            0,             1]])

EXTRINSIC_PARAMETERS = np.array([1, 0, 0, -.1])  # Eye fish and perspective distortion.


class progress_bar:
    def __init__(self, *iterable, append='', length=False, bar_size=100):
        """Prints a nice progress bar while being iterated. Inputs an iterable or the same arguments as range."""
        if not iterable: raise ValueError()
        iterable = range(*iterable) if isinstance(iterable[0], int) else iterable[0]
        self.iterable, self.append, self.start, self.len, self.i, self.r, self.bar_size = iter(iterable), append, False, length if length else len(iterable), 0, None, bar_size

    def __iter__(self): return self

    def __next__(self):
        from time import monotonic
        now = monotonic()
        self.start = self.start or now
        took = int(now - self.start)
        if self.i >= self.len:
            print(f"\r{self.i: >6}/{self.len:<} (100%) [{'■' * self.bar_size}]  Took:" + (f'{took // 60: 3}m' if took >= 60 else '    ') + f'{took % 60:3}s  ' + self.append.format(self.r, self.i))
        self.r = next(self.iterable)
        eta = int((self.len - self.i) * (now - self.start) / self.i) if self.i else 0
        done = self.bar_size * self.i / self.len
        print('\r' + f"{self.i: 6}/{self.len:<} ({int(100 * self.i / self.len): 3}%) [{{:·<{self.bar_size + 4}}}]  ".format('■' * int(done) + str(int(10 * (done % 1))))
              + ('ETA:' + (f'{eta // 60: 4}m' if eta >= 60 else '     ') + f'{eta % 60:3}s  ' if eta else '  ') + self.append.format(self.r, self.i), end='')
        self.i += 1
        return self.r


def calibrate_folder(path=r'\\10.10.204.24\scan4d\TENDER\HANDS\01_HANDS_SIN_CALIBRAR/',
                     dest=r'\\10.10.204.24\scan4d\TENDER\HANDS\02_HANDS_CALIBRADAS/',
                     used_path=r'\\10.10.204.24\scan4d\TENDER\HANDS\01_HANDS_SIN_CALIBRAR\Filtradas/',
                     calibration_json=None):
    intrinsic_matrix = INTRINSIC_MATRIX
    extrinsic_parameters = EXTRINSIC_PARAMETERS
    if calibration_json is not None:
        with open(calibration_json, 'r') as file:
            # Load the calibration from the JSON file. It's a valid python dict.
            # If SyntaxError: invalid syntax, it's probably because the file needs commas in the matrix. Add them manually.
            calibration_dict = eval(file.read())
        intrinsic_matrix = np.array(calibration_dict['camera_matrix'])
        extrinsic_parameters = np.array(calibration_dict['distortion_coefficients'])
    
    files = [f for f in os.listdir(path) if f.endswith('.png') and 'undistorted' not in f]
    print(f'Calibrating {len(files)} images from {path} to {dest} and moving original images to {used_path}.')
    for file in progress_bar(files):
        undistorted = cv2.undistort(cv2.imread(os.path.join(path, file)), intrinsic_matrix, extrinsic_parameters)
        
        # Save the undistorted image with a "label" in the name.
        idx = file.find('.')
        cv2.imwrite(os.path.join(dest, file[:idx] + '.undistorted' + file[idx:]), undistorted)
        
        # If used_path is given, move the original image there.
        if used_path is not None and not os.path.exists(os.path.join(used_path, file)):
            os.rename(os.path.join(path, file), os.path.join(used_path, file))
    print('Done with calibration.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='?', default=r'\\10.10.204.24\scan4d\TENDER\HANDS\01_HANDS_SIN_CALIBRAR/',
                        help='Path to the folder containing the PNG images to be calibrated.')
    parser.add_argument('dest', nargs='?', default=r'\\10.10.204.24\scan4d\TENDER\HANDS\02_HANDS_CALIBRADAS/',
                        help='Destination path of the calibrated PNG images.')
    parser.add_argument('used_path', nargs='?', default=r'\\10.10.204.24\scan4d\TENDER\HANDS\01_HANDS_SIN_CALIBRAR\Filtradas/',
                        help='Path to move the original PNG images to.')
    parser.add_argument('--calibration-json', '--calibration_json', default=None,
                        help='Path to the JSON file containing the calibration matrix and distortion coefficients.')
    return parser.parse_args()


if __name__ == '__main__':
    calibrate_folder(**parse_args().__dict__)
