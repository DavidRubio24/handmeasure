"""
This class has all the imformation and functions to create the GUI:
- The original image and .
- The point locations.
- The modified image with the points and the measure linesd drawn.
- The zoom level, i.e. the region where to crop the image.
- The window title.
- The function to draw the points and the measure lines: show_image.
- The function to handle the mouse events: mouse_callback.
- The function to find the closest point to the mouse position: closest_point.
- The function to find the closest edge to the mouse position: closest_edge.
- A function with the event_loop that repeateadly updates the image and reacts to keyboard events.
"""

import cv2
import numpy as np

from measure import mesure_closed, mesure_opened

COLOR_SCHEME_POINTS = [[221, 229, 205], [227, 30, 58], [112, 110, 112], [233, 59, 147], [172, 212, 191], [22, 42, 79], [56, 137, 192], [52, 18, 199], [162, 247, 132], [54, 129, 157], [39, 29, 226], [164, 126, 30], [32, 70, 53], [220, 28, 142], [33, 249, 24], [127, 148, 194], [57, 206, 55], [162, 222, 243], [72, 148, 77], [169, 228, 236], [114, 69, 177], [145, 176, 127], [39, 208, 225], [237, 120, 42], [165, 135, 78], [0, 29, 129], [143, 144, 59], [7, 106, 219], [58, 78, 77], [38, 126, 209], [90, 198, 169], [59, 16, 221], [249, 96, 196], [162, 129, 137], [223, 9, 143], [216, 3, 123], [204, 156, 173], [134, 23, 5], [123, 202, 252], [154, 144, 40], [119, 43, 192], [192, 229, 58], [236, 161, 205], [18, 120, 170], [149, 176, 50], [94, 104, 174], [192, 67, 17], [20, 118, 178], [60, 210, 131], [110, 188, 212]]
COLOR_SCHEME_POINTS = np.array(COLOR_SCHEME_POINTS, np.uint8)

COLOR_SCHEME_MEASURES = {'handBreadthMeta_C_m1_3-C_m1_2': [221, 229, 205], 'handBreadthMeta_perpendicular_hand': [227, 30, 58], 'O_f1DistalL': [112, 110, 112], 'O_f2Tip': [233, 59, 147], 'O_f2DistalR': [172, 212, 191], 'O_f2DistalL': [22, 42, 79], 'O_f2MedialR': [56, 137, 192], 'O_f2MedialL': [52, 18, 199], 'O_f3Tip': [162, 247, 132], 'O_f3DistalR': [54, 129, 157], 'O_f3DistalL': [39, 29, 226], 'O_f3MedialR': [164, 126, 30], 'O_f3MedialL': [32, 70, 53], 'O_f4Tip': [220, 28, 142], 'O_f4DistalR': [33, 249, 24], 'O_f4DistalL': [127, 148, 194], 'O_f4MedialR': [57, 206, 55], 'O_f4MedialL': [162, 222, 243], 'O_f5Tip': [72, 148, 77], 'O_f5DistalR': [169, 228, 236], 'O_f5DistalL': [114, 69, 177], 'O_f5MedialR': [145, 176, 127], 'O_f5MedialL': [39, 208, 225], 'C_f1Tip': [221, 229, 205], 'C_f2Tip': [227, 30, 58], 'C_f3Tip': [112, 110, 112], 'C_f4Tip': [233, 59, 147], 'C_f5Tip': [172, 212, 191], 'C_f1BaseC': [22, 42, 79], 'C_f2BaseC': [56, 137, 192], 'C_f3BaseC': [52, 18, 199], 'C_f4BaseC': [162, 247, 132], 'C_f5BaseC': [54, 129, 157], 'C_f1Defect': [39, 29, 226], 'C_wristBaseC': [164, 126, 30], 'C_palmBaseC': [32, 70, 53], 'C_m1_2': [220, 28, 142], 'C_m1_3': [33, 249, 24], 'handLength': [221, 229, 205], 'palmLength': [227, 30, 58], 'handThumbLength': [112, 110, 112], 'handIndexLength': [233, 59, 147], 'handMidLength': [172, 212, 191], 'handFourLength': [22, 42, 79], 'handLittleLength': [56, 137, 192], 'handLengthCrotch': [52, 18, 199], 'handBreadthMeta_perpendicular_finger3': [162, 247, 132], 'handThumbBreadth': [54, 129, 157], 'handIndexBreadthDistal': [39, 29, 226], 'handMidBreadthDistal': [164, 126, 30], 'handFourBreadthDistal': [32, 70, 53], 'handLittleBreadthDistal': [220, 28, 142], 'handIndexBreadthProx': [33, 249, 24], 'handMidBreadthMid': [127, 148, 194], 'handFourBreadthMid': [57, 206, 55], 'handLittleBreadthMid': [162, 222, 243], 'handThumbLengthDistal': [72, 148, 77], 'handIndexLengthDistal': [169, 228, 236], 'handMidLengthDistal': [114, 69, 177], 'handFourLengthDistal': [145, 176, 127], 'handLittleLengthDistal': [39, 208, 225], 'handIndexLengthMid': [237, 120, 42], 'handMidLengthMid': [165, 135, 78], 'handFourLengthMid': [0, 29, 129], 'handLittleLengthMid': [143, 144, 59]}


class CorrectorGUI:
    def __init__(self, image_path: str, points: np.ndarray, image_path_dst: str):
        self.image_edges = None
        """Image with the detected edges. Used to find the edges of the hand, where the landmarks should be."""
        self.points_original = np.array(points, copy=False)
        """Original points, before any modification. Used to reset the points."""
        self.points = np.array(points, dtype=float, copy=True)
        """Points to be modified by the user."""
        self.crop = (max(0, round(self.points[:, 1].min()) - 100),
                     max(0, round(self.points[:, 0].min()) - 300),
                     round(self.points[:, 1].max()) + 100,
                     round(self.points[:, 0].max()) + 300)
        """Crop to be applied to the image. It's a tuple of (y0, x0, y1, x1)."""

        self.image_path_dst = image_path_dst
        """Path to the image to be corrected. Used to save a JPG showing the corrected points."""
        self.image = cv2.imread(image_path)
        """Original image. It's not modified."""
        self.modified_image = self.image.copy()
        """Image to be shown to the user. It's modified to show the points and the measures."""

        self.title = 'IBV - Corrector de medidas'
        self.moving_point = None
        """Index of the point being moved by the user with the mouse."""
        self.last_point = None
        """Index of the last point moved by the user with the mouse."""
        self.shift = 0
        """Positive if the user has pressed the shift key."""
        
        cv2.namedWindow(self.title, cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(self.title, self.on_mouse)

    def show_image(self):
        """Draws the points and measures and shows the image."""
        self.modified_image[:] = self.image  # Copy.
        radius = 10
        # Draw points
        for (x, y), color in zip(self.points, COLOR_SCHEME_POINTS):
            # Ignore out of bounds points.
            if y >= self.modified_image.shape[0] or x >= self.modified_image.shape[1]:
                continue
            # If the point is negative, it doesn't count: paint it black.
            # If the point has decimals, it was estimated by the model: paint it white.
            # If the point is an integer, it was measured by the user: paint it red.
            if x < 0:
                circle_color = (0, 0, 0)
                x *= -1
                y *= -1
            elif x % 1:
                circle_color = (255, 255, 255)
            else:
                circle_color = (0, 0, 255)
            # Draw a cross at the point surrounded by a circle.
            x, y = round(x), round(y)
            self.modified_image[max(0, y - radius):y + radius + 1, x:x+1] = color
            self.modified_image[y:y+1, max(0, x - radius):x + radius + 1] = color
            cv2.circle(self.modified_image, (x, y), radius, circle_color, 2)

        measures: dict[str, tuple] = {}
        """Dict of measure names to (start, end) points."""
        
        # We check the number of points to know if the hand is closed or opened.
        # This feels wrong. If something changes, it will break.
        if len(self.points) == 15:
            measures = mesure_closed(self.points)
        elif len(self.points) == 23:
            measures = mesure_opened(self.points)

        # Draw measures.
        for name, ((x0, y0), (x1, y1)) in measures.items():
            cv2.line(self.modified_image, (round(abs(x0)), round(abs(y0))), (round(abs(x1)), round(abs(y1))), COLOR_SCHEME_MEASURES[name], 2)

        cv2.imshow(self.title, self.modified_image[self.crop[0]:self.crop[2], self.crop[1]:self.crop[3]])

    def on_mouse(self, event, x, y, flags, *_):
        if event == cv2.EVENT_RBUTTONDOWN:
            self.moving_point = self.closest_point(x + self.crop[1], y + self.crop[0])
            self.last_point = self.moving_point
            # If the shift key is pressed, the point will stick to the closest edge.
            sticky_edges = flags & cv2.EVENT_FLAG_SHIFTKEY
            self.points[self.moving_point, :2] = self.closest_edge(x + self.crop[1], y + self.crop[0], sticky_edges)
            self.show_image()
        elif self.moving_point is not None and event == cv2.EVENT_MOUSEMOVE:
            # If the shift key is pressed, the point will stick to the closest edge.
            sticky_edges = flags & cv2.EVENT_FLAG_SHIFTKEY
            self.points[self.moving_point, :2] = self.closest_edge(x + self.crop[1], y + self.crop[0], sticky_edges)
            self.show_image()
        elif event == cv2.EVENT_RBUTTONUP:
            self.moving_point = None
            self.show_image()

    def closest_edge(self, x, y, sticky_edges=False, radius=100):
        """Returns the closest edge to the given point if sticky_edges."""
        if not sticky_edges:
            return x, y

        # Only compute the edges the first time they are needed.
        if self.image_edges is None:
            # Blur the red channel (the most representative for the hand) and detect its edges.
            # TODO: Those parameters have been chosen empiricaly for our scanner.
            #       For other images they may not work as well.
            #       Delft probably needs a bigger blur (~20 instead of 11).
            image_blured = cv2.GaussianBlur(self.image[..., 2], (11, 11), 0)
            self.image_edges = cv2.Canny(image_blured, 1000, 1100, apertureSize=5)

        # If the point is already on an edge (a non-null image_edges pixel), return it.
        if self.image_edges[y, x]:
            return x, y

        # Only look for edges in a small area around the point.
        min_x, min_y = max(0, x - radius), max(0, y - radius)
        area = self.image_edges[min_y:y + radius,
                                min_x:x + radius]

        indices = np.argwhere(area)  # area indices are coordinates and start at (x, y) - (radius, radius)
        if len(indices) == 0:
            # No edges in the area, return the current point.
            return x, y
        
        distances = np.linalg.norm(indices - (y - min_y, x - min_x), axis=1)
        closest_index = np.argmin(distances)

        return indices[closest_index, 1] + min_x, indices[closest_index, 0] + min_y

    def closest_point(self, x, y):
        """Returns the index of the closest keypoint to the given coordinates."""
        distances = (self.points[..., 0] - x) ** 2 + (self.points[..., 1] - y) ** 2
        return np.argmin(distances)

    def event_loop(self):
        update_image = True
        """Whether the image should be updated in the next iteration with the new point positions and measures."""
        while True:
            if update_image:
                self.show_image()
                update_image = False
            else:
                cv2.imshow(self.title, self.modified_image[self.crop[0]:self.crop[2], self.crop[1]:self.crop[3]])
            
            # Wait for a key to be pressed.
            key_pressed = cv2.waitKey()
            
            if key_pressed == 27:  # Esc
                # Reset points to the original ones.
                self.points = self.points_original.copy()
                update_image = True
            elif key_pressed == 8:  # Backspace
                # End correction without saving anything.
                return None
            elif key_pressed in [32, 13, ord('g')]:  # Space, enter or 'g'
                # Save the points and end correction.
                cv2.imwrite(self.image_path_dst[:-4] + '.measures.jpg', self.modified_image)
                return self.points if np.any(self.points != self.points_original) else None
            elif key_pressed == ord('-'):
                # Zoom out. Add 10 pixels to each side.
                x0, y0, x1, y1 = self.crop
                crop = (max(x0 - 10, 0), max(y0 - 10, 0), min(x1 + 10, self.image.shape[0] - 1), min(y1 + 10, self.image.shape[1] - 1))
                self.crop = tuple(map(round, crop))
            elif key_pressed == ord('+'):
                # Zoom in. Remove 10 pixels from each side.
                x0, y0, x1, y1 = self.crop
                x0, y0, x1, y1 = (x0 + 10, y0 + 10, x1 - 10, y1 - 10)
                crop = (min(x0, x1 - 11), min(y0, y1 - 11), max(x1, x0 + 11), max(y1, y0 + 11))
                self.crop = tuple(map(round, crop))
            elif key_pressed == 16:  # Shift
                self.shift = 2
                # It will inmediately be reduced to 1.
                # Any additional key pressed will reduce it to 0.
                # So only if Supr is pressed immediately after, it will have any effect.
            elif key_pressed == 46 and self.shift > 0:  # (Shift +) Supr
                # Delete the last point moved. Negative points are considered invalid.
                self.points[self.last_point] *= -1
                update_image = True
            self.shift = max(0, self.shift - 1)
