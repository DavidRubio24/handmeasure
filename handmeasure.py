"""
Process each PNG image in the given folder and generates a JSON file with the landmarks and the distances between them.

PNGs must have either 'close' or 'M1' in their name to be considered closed
and 'open' or 'M2' in their name to be considered opened.
If not, they will be ignored.

Each type of hand (closed or opened) has a different set of landmarks and distances.

The JSON file will be saved in the same folder as the PNG with the same name but with a .json extension.
The content of the JSON will be both a JSON valid file and a python dict.

Usage:
    python handmeasure.py <path> [--auto]

    <path> is the path to the folder containing the images.
    --auto: if present, the program will process all the images in the folder without human correction.
"""

import os
import argparse

import cv2
import numpy as np

from constants import points_interest_closed, points_interest_opened
from GUI import CorrectorGUI
from measure import compute_distances, mesure_closed, mesure_opened
# There's a conditional import: from landmarks import get_landmarks
# It imports mediapipe which takea a lot of time to load. So it's imported only when needed.

INPUT_FILE_FORMATS = ('.png', )


def main(path=r'\\10.10.204.24\scan4d\TENDER\HANDS\02_HANDS_CALIBRADAS/', auto=False,
         save_path=r'\\10.10.204.24\scan4d\TENDER\HANDS\02_HANDS_CALIBRADAS\REVISADAS/'):
    for file in os.listdir(path):
        if not file.endswith(INPUT_FILE_FORMATS):
            continue

        file_dst = os.path.join(save_path, file)
        file = os.path.join(path, file)

        # Find out if the file is closed or opened.
        if 'close' in file.lower() or 'M1' in file.upper():
            closed = True
        elif 'open' in file.lower() or 'M2' in file.upper():
            closed = False
        else:
            print(f'{file} no es ni abierto ni cerrado. Se ignora.')
            continue

        points_interest = points_interest_closed if closed else points_interest_opened

        # Get the landmarks from the corresponding JSON file
        # if exists or generate them automatically if not.
        basename, extension = os.path.splitext(file_dst)
        json = basename + '.json'
        if os.path.exists(json):
            print(f'Cargando puntos de {json}...')
            with open(json, 'r') as json_file:
                landmarks_dict = eval(json_file.read())
            landmarks = np.array([landmarks_dict[point] for point in points_interest])
            updated_landmarks = False
        else:
            print(f'{file} no tiene landmarks. Se generarán automaticamente.')
            image = cv2.imread(file)
            if image is None:
                print(f'No se puede leer {file}.')
                continue
            image_rgb = image[..., ::-1]
            from landmarks import get_landmarks
            landmarks = get_landmarks(image_rgb, closed) + .001
            updated_landmarks = True

        if not auto:
            print(f'Corrige landmarks de {file}...')
            corrector_gui = CorrectorGUI(file, landmarks, file_dst)
            landmarks_updated = corrector_gui.event_loop()
            cv2.destroyWindow(corrector_gui.title)
            if landmarks_updated is not None:
                updated_landmarks = True
                landmarks = landmarks_updated

        if updated_landmarks:
            print(f'Guardando landmarks de {file} actualizados.')
            json_content = {point: landmarks[i].tolist() for i, point in enumerate(points_interest)}
            json_content |= compute_distances(mesure_closed(landmarks) if closed else mesure_opened(landmarks))
            with open(json, 'w') as json_file:
                json_file.write(str(json_content)
                                # Reformat the dict into a JSON.
                                .replace(", '", ",\n'")
                                .replace("{'", "{\n'")
                                .replace("}", "\n}")
                                .replace("': [", "':\t[")
                                .replace("'", '"')
                                )
            os.rename(file, file_dst)
        else:
            print(f'No se han actualizado los landmarks de {file}.')

    print('Fin.')


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Automatically estimates the landmarks of the hands in the PNG images in the given folder. "
        "Let's a user correct them (unless --auto is specified). Computes the distances between the landmarks "
        "and saves them in a JSON file with the same name as the PNG image."
    )
    parser.add_argument('path', nargs='?', default=r'\\10.10.204.24\scan4d\TENDER\HANDS\02_HANDS_CALIBRADAS/',
                        help='Path to the folder containing the PNG images to be processed.')
    parser.add_argument('save_path', nargs='?', default=r'\\10.10.204.24\scan4d\TENDER\HANDS\02_HANDS_CALIBRADAS\REVISADAS/',
                        help='Path to the folder to save move everything after finishing each image.')
    parser.add_argument('--auto', action='store_true', default=False,
                        help='Generate the JSONs with the landmarks without human corrections. (Default: False)')
    return parser.parse_args()


if __name__ == '__main__':
    main(**parse_args().__dict__)
