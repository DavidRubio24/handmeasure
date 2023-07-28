"""
Process each PNG image in the given folder and generates a JSON and JPG file with the landmarks and the distances between them.

PNGs must have either 'close' or 'M1' in their name to be considered closed
and 'open' or 'M2' in their name to be considered opened.
If not, they will be ignored.

Each type of hand (closed or opened) has a different set of landmarks and distances.

The JSON file will be saved in the same folder as the PNG with the same name but with a .json extension.
The content of the JSON will be both a JSON valid file and a python dict.

Usage:
    python handmeasure.py <path> [--auto] [--pixel-size <pixel_size>]

    <path> is the path to the folder containing the images.
    --auto: if present, the program will process all the images in the folder without human correction.
    --pixel-size: the size of the pixels in mm. Default: 1/12.36 (the size of the pixels in our scanner).
"""

import os
import argparse

import cv2
import numpy as np

from constants import points_interest_closed, points_interest_opened
from GUI import CorrectorGUI
from measure import compute_distances, mesure_closed, mesure_opened
# There's a conditional import: from landmarks import get_landmarks
# It imports mediapipe which takes a lot of time to load. So it's imported only when needed, i.e.,
# when the landmarks are not found in a previously generated JSON file.

INPUT_FILE_FORMATS = ('.png', )


def main(path=r'\\10.10.204.24\scan4d\TENDER\HANDS\02_HANDS_CALIBRADAS/',
         save_path=r'\\10.10.204.24\scan4d\TENDER\HANDS\02_HANDS_CALIBRADAS\REVISADAS/',
         auto=False,  # Don't ask for user input, just use the estimation based on MediaPipe.
                      # Useful to genate the JSON files from one computer and checking them
                      # in another computer that can't run MediaPipe.
         pixel_size=1/12.36,  # This value doesn't usually change unless the scanner is modified.
                              # But we periodically check it, measuring the contour ruller in the scans (I used GIMP).
         ):
    for file in os.listdir(path):
        if not file.endswith(INPUT_FILE_FORMATS):
            # Not the right file format. Skip this file.
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

        # Get the landmarks from the corresponding JSON file if exists in the destination folder
        # or generate them automatically if not.
        basename, extension = os.path.splitext(file_dst)
        json = basename + '.json'
        save_landmarks_in_json = False
        if os.path.exists(json):
            print(f'Cargando puntos de {json}...')
            with open(json, 'r') as json_file:
                # Our JSONs are valid python dicts (no use of true, false or null).
                landmarks_dict = eval(json_file.read())
            # Take only the points of interest as an array (ignore the distances, date and pixel size).
            landmarks = np.array([landmarks_dict[point] for point in points_interest])
        else:
            print(f'{file} no tiene landmarks. Se generarán automaticamente.')
            image = cv2.imread(file)
            if image is None:
                print(f'No se puede leer {file}.')
                continue
            image_rgb = image[..., ::-1]
            from landmarks import get_landmarks  # The First time takes a while to load MediaPipe.
            landmarks = get_landmarks(image_rgb, closed)
            
            if landmarks is None:
                print(f'No se ha podido detectar la mano en {file}.')
                continue
            
            # Add an infinitesimal amount to know that the landmarks have been generated automatically.
            # We use this to paint the landmarks in the GUI in a different color to inform the user.
            landmarks += .001
            
            save_landmarks_in_json = True

        # Show the landmarks in the GUI and let the user correct them if not auto.
        if not auto:
            print(f'Corrige landmarks de {file}...')
            # Create an objet with all the information needed to show the GUI.
            corrector_gui = CorrectorGUI(file, landmarks, file_dst)
            # Run the GUI and wait for the user to be done with this image.
            landmarks_updated = corrector_gui.event_loop()
            cv2.destroyWindow(corrector_gui.title)
            if landmarks_updated is not None:
                save_landmarks_in_json = True
                landmarks = landmarks_updated

        if save_landmarks_in_json:
            print(f'Guardando landmarks de {file} actualizados.')
            json_content = {point: landmarks[i].tolist() for i, point in enumerate(points_interest)}
            json_content['pixel_size'] = pixel_size
            
            # If the date is in the filename, add it to the json file.
            if len(file) >= 13 and file[-13] == '.' and file[-12:-4].isdigit():
                date = file[-12:-4]
                json_content['capture_date'] = date[:4] + '-' + date[4:6] + '-' + date[6:]
            
            # Firs compute, for each distance the start and end points in pixel coordinates.
            # Then compute the distances in mm.
            # This is done in two steps, so we can show the lines representing the distances in the GUI.
            pixel_positions = mesure_closed(landmarks) if closed else mesure_opened(landmarks)
            json_content |= compute_distances(pixel_positions, pixel_size)
            
            # Save the JSON file. Start with a str representation of the dict and reformat it.
            with open(json, 'w') as json_file:
                json_file.write(str(json_content)
                                # Reformat the dict into a (pretty) JSON.
                                .replace(", '", ",\n'")
                                .replace("{'", "{\n'")
                                .replace("}", "\n}")
                                .replace("': [", "':\t[")
                                .replace("'", '"')
                                )
            
            # Move the image to the destination folder unless it's already there.
            if os.path.exists(file_dst):
                response = input(f'¿Sobreescribir {file_dst} con {file}? ([s]/n) ')
                if response.strip().lower() not in ('n', 'no', 'not', 'non', 'na', 'nah', 'nay', 'nein'):
                    os.replace(file, file_dst)
                else:
                    print(f'No se ha movido {file} a {file_dst}.')
            # In auto mode, leave the image in the original folder so that the user can check it.
            elif not auto:
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
    parser.add_argument('--pixel-size', '--pixel_size', type=float, default=1/12.36,
                        help='Pixel size in mm. (Default: 1/12.36, the size of the pixels in our scanner')
    
    return parser.parse_args()


if __name__ == '__main__':
    main(**parse_args().__dict__)
