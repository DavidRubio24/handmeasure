"""
Here the landmarks are estimated from the image.

The MediaPipe Hand landmarks are used as a starting point.
Their relative positions are used to determine a line in the image that probably goes through our landmark.
Along that line, the edge of the hand is searched for and used as our landmark.
"""

import numpy as np
from mediapipe.python.solutions.hands import Hands

from constants import *


def get_landmarks(image_rgb: np.ndarray, closed: bool, detector=Hands(static_image_mode=True, max_num_hands=1)):
    """Get the pixel coordinates of the landmarks of the hand in the image."""
    results = detector.process(image_rgb)

    if results is None or results.multi_hand_landmarks is None:
        return None

    landmarks = np.array([(l.x, l.y) for l in results.multi_hand_landmarks[0].landmark])
    landmarks[:, 0] *= image_rgb.shape[1]
    landmarks[:, 1] *= image_rgb.shape[0]

    return get_landmarks_closed(image_rgb, landmarks) if closed else get_landmarks_opened(image_rgb, landmarks)


def get_landmarks_opened(image, landmarks_mediapipe: np.ndarray):
    lmk_mp = np.round(landmarks_mediapipe)
    """MediaPipe Hand landmarks."""

    lmk = np.zeros((len(points_interest_opened), 2), int)
    """Our landmarks."""

    O_f1Tip = 0; O_f1DistalR = 1; O_f1DistalL = 2
    O_f2Tip = 3; O_f2DistalR = 4; O_f2DistalL = 5; O_f2MedialR = 6; O_f2MedialL = 7
    O_f3Tip = 8; O_f3DistalR = 9; O_f3DistalL = 10; O_f3MedialR = 11; O_f3MedialL = 12
    O_f4Tip = 13; O_f4DistalR = 14; O_f4DistalL = 15; O_f4MedialR = 16; O_f4MedialL = 17
    O_f5Tip = 18; O_f5DistalR = 19; O_f5DistalL = 20; O_f5MedialR = 21; O_f5MedialL = 22

    # THUMB
    lmk[O_f1Tip] = get_line_edge(image, lmk_mp[THUMB_TIP], 2 * lmk_mp[THUMB_TIP] - lmk_mp[THUMB_IP])
    finger_direction = lmk_mp[THUMB_TIP] - lmk_mp[THUMB_MCP]
    lmk[O_f1DistalR] = get_line_edge(image, lmk_mp[THUMB_IP], direction=[-finger_direction[1], finger_direction[0]])
    lmk[O_f1DistalL] = get_line_edge(image, lmk_mp[THUMB_IP], direction=[finger_direction[1], -finger_direction[0]])

    # INDEX
    lmk[O_f2Tip] = get_line_edge(image, lmk_mp[INDEX_FINGER_TIP], 2 * lmk_mp[INDEX_FINGER_TIP] - lmk_mp[INDEX_FINGER_PIP])
    finger_direction = lmk_mp[INDEX_FINGER_TIP] - lmk_mp[INDEX_FINGER_PIP]
    lmk[O_f2DistalR] = get_line_edge(image, lmk_mp[INDEX_FINGER_DIP], direction=[-finger_direction[1], finger_direction[0]])
    lmk[O_f2DistalL] = get_line_edge(image, lmk_mp[INDEX_FINGER_DIP], direction=[finger_direction[1], -finger_direction[0]])
    finger_direction = lmk_mp[INDEX_FINGER_DIP] - lmk_mp[INDEX_FINGER_MCP]
    lmk[O_f2MedialR] = get_line_edge(image, lmk_mp[INDEX_FINGER_PIP], direction=[-finger_direction[1], finger_direction[0]])
    lmk[O_f2MedialL] = get_line_edge(image, lmk_mp[INDEX_FINGER_PIP], direction=[finger_direction[1], -finger_direction[0]])

    # MIDDLE
    lmk[O_f3Tip] = get_line_edge(image, lmk_mp[MIDDLE_FINGER_TIP], 2 * lmk_mp[MIDDLE_FINGER_TIP] - lmk_mp[MIDDLE_FINGER_PIP])
    finger_direction = lmk_mp[MIDDLE_FINGER_TIP] - lmk_mp[MIDDLE_FINGER_PIP]
    lmk[O_f3DistalR] = get_line_edge(image, lmk_mp[MIDDLE_FINGER_DIP], direction=[-finger_direction[1], finger_direction[0]])
    lmk[O_f3DistalL] = get_line_edge(image, lmk_mp[MIDDLE_FINGER_DIP], direction=[finger_direction[1], -finger_direction[0]])
    finger_direction = lmk_mp[MIDDLE_FINGER_DIP] - lmk_mp[MIDDLE_FINGER_MCP]
    lmk[O_f3MedialR] = get_line_edge(image, lmk_mp[MIDDLE_FINGER_PIP], direction=[-finger_direction[1], finger_direction[0]])
    lmk[O_f3MedialL] = get_line_edge(image, lmk_mp[MIDDLE_FINGER_PIP], direction=[finger_direction[1], -finger_direction[0]])

    # RING
    lmk[O_f4Tip] = get_line_edge(image, lmk_mp[RING_FINGER_TIP], 2 * lmk_mp[RING_FINGER_TIP] - lmk_mp[RING_FINGER_PIP])
    finger_direction = lmk_mp[RING_FINGER_TIP] - lmk_mp[RING_FINGER_PIP]
    lmk[O_f4DistalR] = get_line_edge(image, lmk_mp[RING_FINGER_DIP], direction=[-finger_direction[1], finger_direction[0]])
    lmk[O_f4DistalL] = get_line_edge(image, lmk_mp[RING_FINGER_DIP], direction=[finger_direction[1], -finger_direction[0]])
    finger_direction = lmk_mp[RING_FINGER_DIP] - lmk_mp[RING_FINGER_MCP]
    lmk[O_f4MedialR] = get_line_edge(image, lmk_mp[RING_FINGER_PIP], direction=[-finger_direction[1], finger_direction[0]])
    lmk[O_f4MedialL] = get_line_edge(image, lmk_mp[RING_FINGER_PIP], direction=[finger_direction[1], -finger_direction[0]])

    # PINKY
    lmk[O_f5Tip] = get_line_edge(image, lmk_mp[PINKY_TIP], 2 * lmk_mp[PINKY_TIP] - lmk_mp[PINKY_PIP])
    finger_direction = lmk_mp[PINKY_TIP] - lmk_mp[PINKY_PIP]
    lmk[O_f5DistalR] = get_line_edge(image, lmk_mp[PINKY_DIP], direction=[-finger_direction[1], finger_direction[0]])
    lmk[O_f5DistalL] = get_line_edge(image, lmk_mp[PINKY_DIP], direction=[finger_direction[1], -finger_direction[0]])
    finger_direction = lmk_mp[PINKY_DIP] - lmk_mp[PINKY_MCP]
    lmk[O_f5MedialR] = get_line_edge(image, lmk_mp[PINKY_PIP], direction=[-finger_direction[1], finger_direction[0]])
    lmk[O_f5MedialL] = get_line_edge(image, lmk_mp[PINKY_PIP], direction=[finger_direction[1], -finger_direction[0]])

    return lmk


def get_landmarks_closed(image, lmk_mp: np.ndarray):
    lmk_mp = np.round(lmk_mp)
    """MediaPipe Hand landmarks."""

    lmk = np.zeros((len(points_interest_closed), 2), int)
    """Our landmarks."""

    C_f1Tip = 0; C_f1BaseC = 5; C_f1Defect = 10
    C_f2Tip = 1; C_f2BaseC = 6
    C_f3Tip = 2; C_f3BaseC = 7
    C_f4Tip = 3; C_f4BaseC = 8
    C_f5Tip = 4; C_f5BaseC = 9
    C_wristBaseC = 11; C_palmBaseC = 12
    C_m1_2 = 13; C_m1_3 = 14

    # THUMB
    lmk[C_f1Tip] = get_line_edge(image, lmk_mp[THUMB_TIP], 2 * lmk_mp[THUMB_TIP] - lmk_mp[THUMB_IP])
    lmk[C_f1BaseC] = lmk_mp[THUMB_MCP] * .95 + lmk_mp[THUMB_CMC] * .05
    lmk[C_f1Defect] = lmk_mp[THUMB_MCP] * .7 + lmk_mp[INDEX_FINGER_MCP] * .3

    # INDEX
    lmk[C_f2Tip] = get_line_edge(image, lmk_mp[INDEX_FINGER_TIP], 2 * lmk_mp[INDEX_FINGER_TIP] - lmk_mp[INDEX_FINGER_PIP])
    lmk[C_f2BaseC] = lmk_mp[INDEX_FINGER_MCP] * (2 / 3) + lmk_mp[INDEX_FINGER_PIP] / 3

    # MIDDLE
    lmk[C_f3Tip] = get_line_edge(image, lmk_mp[MIDDLE_FINGER_TIP], 2 * lmk_mp[MIDDLE_FINGER_TIP] - lmk_mp[MIDDLE_FINGER_PIP])
    lmk[C_f3BaseC] = lmk_mp[MIDDLE_FINGER_MCP] * (2 / 3) + lmk_mp[MIDDLE_FINGER_PIP] / 3

    # RING
    lmk[C_f4Tip] = get_line_edge(image, lmk_mp[RING_FINGER_TIP], 2 * lmk_mp[RING_FINGER_TIP] - lmk_mp[RING_FINGER_PIP])
    lmk[C_f4BaseC] = lmk_mp[RING_FINGER_MCP] * (2 / 3) + lmk_mp[RING_FINGER_PIP] / 3

    # PINKY
    lmk[C_f5Tip] = get_line_edge(image, lmk_mp[PINKY_TIP], 2 * lmk_mp[PINKY_TIP] - lmk_mp[PINKY_PIP])
    lmk[C_f5BaseC] = lmk_mp[PINKY_MCP] * (2 / 3) + lmk_mp[PINKY_PIP] / 3

    lmk[C_wristBaseC] = lmk_mp[WRIST] * 1.1 - lmk_mp[MIDDLE_FINGER_MCP] * .1
    lmk[C_palmBaseC] = lmk_mp[WRIST]

    lmk[C_m1_2] = get_line_edge(image, lmk_mp[INDEX_FINGER_MCP], direction_scale=1,
                                direction=lmk_mp[INDEX_FINGER_MCP] - lmk_mp[MIDDLE_FINGER_MCP], )
    lmk[C_m1_3] = get_line_edge(image, lmk_mp[PINKY_MCP], direction_scale=1,
                                direction=lmk_mp[PINKY_MCP] - lmk_mp[RING_FINGER_MCP], )

    return lmk


def get_line_edge(image, point1: np.ndarray, point2=None, direction=None, direction_scale=1/3):
    """
    Get the location of the edge of the hand in the continuation of the line between the first and second point or
    from the first point in the direction of the direction vector.
    """
    # Make sure the point is inside the image.
    if not 0 <= point1[0] < image.shape[1] or not 0 <= point1[1] < image.shape[0]:
        # TODO: This should not happen. But if it does, this is not the right way to handle it.
        #       At least we should return the closest point that is located in the image.
        return point1

    # Get the second point from the direction if it is not given.
    point2 = point2 if point2 is not None else point1 + np.array(direction) * direction_scale

    # Get the locations of the line that goes from the first to the second point.
    line_length = int(np.ceil(np.linalg.norm(point1 - point2)))
    line_locations = np.linspace(point1, point2, line_length, dtype=int)

    # Crop the indices to the image.
    line_locations = line_locations[(line_locations[:, 0] >= 0) & (line_locations[:, 0] < image.shape[1])]
    line_locations = line_locations[(line_locations[:, 1] >= 0) & (line_locations[:, 1] < image.shape[0])]

    # Get the color of each pixel in the line.
    line = np.array([image[y, x] for x, y in line_locations], dtype=int)

    # Compute the color changes along the line.
    kernel = np.array([-2, -1, 0, 0, 1, 2])
    kernel_offset = (kernel.shape[0] - 1) // 2
    change_abs = [np.convolve(line[:, channel], kernel, 'valid') for channel in range(line.shape[1])]
    change_rate = np.linalg.norm(change_abs, axis=0)

    # Find the N most significant color changes.
    N = 10
    indices = np.argpartition(change_rate, kth=len(change_rate) - N)[-N:]
    # Order them by their change rate.
    indices = indices[np.argsort(change_rate[indices])]
    # indices = [indices[0]] + [index for i, index in enumerate(indices[1:], start=1) if max(abs(indices[:i] - index)) > 4]  # Exclude the changes that are too close to each other.

    # Exclude the changes that get more similar to the color of the start of the line.
    def before(index): return max(0, index - kernel_offset)
    def after(index): return min(len(line), index + kernel_offset)
    indices = ([index for index in indices
                if np.linalg.norm(line[before(index)] - line[0])  # Similarity to the start of the line.
                < np.linalg.norm(line[after(index)] - line[0]) + 5]  # Similarity to the end of the line with a margin.
               or indices)  # If all changes are excluded, use all of them anyway.
    edge = indices[-1]  # Keep the most significant change.
    edge_location = line_locations[edge + kernel_offset + 1]

    return edge_location
