"""
For eash measurment we have a pair of points.

There is a function for each hand pose (opened or closed) that returns a dict of measeure names
to their corresponting pairs of points.

There's another function that takes the pairs of points and returns a dict of measeure names
to their corresponting distances, scaled by pixel size.

Negative signs in point coordinates mean that the measurement is invalid.
We still compute its value and return the distance as a negative number.
"""
import numpy as np

from constants import points_interest_closed, points_interest_opened


def mean_sign(x, y):
    """Return the mean of the absolute values of x and y with a negative sign if something is negative."""
    sign = 1 if np.all(x >= 0) and np.all(y >= 0) else -1
    return sign * np.mean([abs(x), abs(y)], axis=0)


def mesure_opened(points: np.ndarray) -> dict[str, tuple]:
    """Return a dict of measure names to their (start, end) points."""
    points = {name: np.asarray(point) for name, point in zip(points_interest_opened, points)}

    distance = {'handThumbBreadth':        (points['O_f1DistalR'], points['O_f1DistalL']),
                'handIndexBreadthDistal':  (points['O_f2DistalR'], points['O_f2DistalL']),
                'handMidBreadthDistal':    (points['O_f3DistalR'], points['O_f3DistalL']),
                'handFourBreadthDistal':   (points['O_f4DistalR'], points['O_f4DistalL']),
                'handLittleBreadthDistal': (points['O_f5DistalR'], points['O_f5DistalL']),

                'handIndexBreadthProx':    (points['O_f2MedialR'], points['O_f2MedialL']),
                'handMidBreadthMid':       (points['O_f3MedialR'], points['O_f3MedialL']),
                'handFourBreadthMid':      (points['O_f4MedialR'], points['O_f4MedialL']),
                'handLittleBreadthMid':    (points['O_f5MedialR'], points['O_f5MedialL']),

                'handThumbLengthDistal':   (points['O_f1Tip'], mean_sign(points['O_f1DistalL'], points['O_f1DistalR'])),
                'handIndexLengthDistal':   (points['O_f2Tip'], mean_sign(points['O_f2DistalL'], points['O_f2DistalR'])),
                'handMidLengthDistal':     (points['O_f3Tip'], mean_sign(points['O_f3DistalL'], points['O_f3DistalR'])),
                'handFourLengthDistal':    (points['O_f4Tip'], mean_sign(points['O_f4DistalL'], points['O_f4DistalR'])),
                'handLittleLengthDistal':  (points['O_f5Tip'], mean_sign(points['O_f5DistalL'], points['O_f5DistalR'])),

                'handIndexLengthMid':     (mean_sign(points['O_f2DistalL'], points['O_f2DistalR']),
                                           mean_sign(points['O_f2MedialL'], points['O_f2MedialR'])),
                'handMidLengthMid':       (mean_sign(points['O_f3DistalL'], points['O_f3DistalR']),
                                           mean_sign(points['O_f3MedialL'], points['O_f3MedialR'])),
                'handFourLengthMid':      (mean_sign(points['O_f4DistalL'], points['O_f4DistalR']),
                                           mean_sign(points['O_f4MedialL'], points['O_f4MedialR'])),
                'handLittleLengthMid':    (mean_sign(points['O_f5DistalL'], points['O_f5DistalR']),
                                           mean_sign(points['O_f5MedialL'], points['O_f5MedialR'])),
                }

    return distance


def mesure_closed(points: np.ndarray) -> dict[str, tuple]:
    """Return a dict of measure names to their (start, end) points."""
    points = {name: np.asarray(point).astype(float, copy=False) for name, point in zip(points_interest_closed, points)}

    distance = {'handLength':          (points['C_f3Tip'],   points['C_wristBaseC']),
                'palmLength':          (points['C_f3BaseC'], points['C_palmBaseC']),
                'handThumbLength':     (points['C_f1Tip'],   points['C_f1BaseC']),
                'handIndexLength':     (points['C_f2Tip'],   points['C_f2BaseC']),
                'handMidLength':       (points['C_f3Tip'],   points['C_f3BaseC']),
                'handFourLength':      (points['C_f4Tip'],   points['C_f4BaseC']),
                'handLittleLength':    (points['C_f5Tip'],   points['C_f5BaseC']),
                # 'handBreadthMeta_C_m1_3-C_m1_2': (points['C_m1_2'], points['C_m1_3']),
                }

    # handLengthCrotch parallel to middle finger.
    direction = abs(points['C_f3Tip']) - abs(points['C_f3BaseC'])
    direction /= np.linalg.norm(direction)

    handLengthCrotch = np.dot(abs(points['C_f3Tip']) - abs(points['C_f1Defect']), direction) * direction + abs(points['C_f1Defect'])
    distance['handLengthCrotch'] = (handLengthCrotch, points['C_f1Defect'])
    if np.any(points['C_f3Tip'] < 0) or np.any(points['C_f3BaseC'] < 0) or np.any(points['C_f1Defect'] < 0):
        distance['handLengthCrotch'] = tuple(np.array(distance['handLengthCrotch']) * -1)

    # handBreadthMeta perpendicular to middle finger.
    # direction[:] = -direction[1], direction[0]
    # handBreadthMeta = np.dot(points['C_m1_2'] - points['C_m1_3'], direction) * direction + points['C_m1_3']
    # distance['handBreadthMeta_perpendicular_finger3'] = (handBreadthMeta, points['C_m1_3'])
    direction = abs(points['C_f3Tip']) - abs(points['C_wristBaseC'])
    direction /= np.linalg.norm(direction)
    direction[:] = -direction[1], direction[0]
    handBreadthMeta = np.dot(abs(points['C_m1_2']) - abs(points['C_m1_3']), direction) * direction + abs(points['C_m1_3'])
    distance['handBreadthMeta_perpendicular_hand'] = (handBreadthMeta, abs(points['C_m1_3']))
    if np.any(points['C_m1_2'] < 0) or np.any(points['C_m1_3'] < 0) or np.any(points['C_f3Tip'] < 0) or np.any(points['C_wristBaseC'] < 0):
        distance['handBreadthMeta_perpendicular_hand'] = tuple(np.array(distance['handBreadthMeta_perpendicular_hand']) * -1)

    return distance


def compute_distances(points: dict, pixel_size: float = 138 / 1711.1):
    """
    Given a dictionary of names to pairs of points, compute the distance between the pairs of points.

    If the sign of a point is negative, the distance is negative.

    Scale the distance by the given factor (to account for pixel size).
    """
    distances = {name: np.linalg.norm(abs(p1) - abs(p0)) * pixel_size * (1 if np.all(p0 >= 0) and np.all(p1 >= 0) else -1)
                 for name, (p0, p1) in points.items()}
    return distances
