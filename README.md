# Handmesure

Utilities to locate points of interest in images of scanned hands to mesure the hand.


This project aims to detect points in two different poses: opened and closed.
Each pose has its own points to locate: 23 for the opened hands and 15 for the closed ones.

As of right now (2023/4/12) the strategy is as follows:
1. Use MediaPipe Hands to detect hand keypoints.
2. Use those keypoints and the edges of the hand as a reference to find the landmarks.
3. Show a human the points, so she can modify them if they're wrong.

The calibrate.py script is independent and is here for convinience. 


### Install

It needs conda (https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe).

From the Anaconda prompt, create a new environment with the numpy and opencv libraries:

```conda create -n p10 python=3.10 numpy opencv```

Sometimes it's necessary to close and open the promt again.

Activate the environment:

```conda activate p10```

Install mediapipe with pip:

```pip install mediapipe```

it will install numpy and opencv as dependencies,
but pip's opencv doesn't use the QT backend (that allows to zoom by scrolling).

Remove pip's opencv:

```pip uninstall opencv-contrib-python```


### Usage

From the conda prompt, activate the environment:

```conda activate p10```

Change to the directory where the project is located, for example:

```
D:
cd handmeasure
```

To calibrate the images:

```python calibrate.py folder/with/uncalibrated/images folder/to/save/calibrated/images folder/to/move/uncalibrated/images```

For some unknown reason, using calibrate.py from the command line doesn't always work.
So it has to be run from the python interpreter or, equivalently,
from the command line with the -c option:

```python -c "import calibrate; calibrate.calibrate_folder(r'folder/with/uncalibrated/images/', r'folder/to/save/calibrated/images/', r'folder/to/move/uncalibrated/images/')"```

To measure the hands:

```python handmeasure.py path/to/folder/with/images```

Again, it has to be run from the python interpreter or from the command line with the -c option:

```python -c "import handmeasure; handmeasure.main(r'path/to/folder/with/images')"```

It will show the images one by one with the estimated keypoint locations and the measurements between them.

The user can move the points by right-clicking: when the right mouse button is pressed,
the closest point will be moved to the mouse position,
so there is no need of dragging the point (but it can be done).

By pressing shift and right-clicking,
the closest point to the mose will be moved to the closest edge to the mouse.
This edge is not guaranteed to be where the user expects, and it should be checked.
_(Implementation details: the edge is found with the Canny edge detector over the blured red channel
(the one where the hand contrasts the most), using empirically determined parameters.
The edge is only looked for in a small area around the mouse.)_

Pressing Esc resets the points to the original position.

Pressing Enter, g, or the space bar saves the points and moves to the next image.

Pressing Backspace skips this image without saving anything.

Mouse scrolling zooms in and out. This is opencv's zoom. Only available when it uses QT.

Pressing + and - zooms in and out. This is a hard zoom (it crops the image).

Pressing Shift + Delete 'deletes' the last point that was moved.
It doesn't actually delete it,
but it paints it in black and stores its coordinates and related measurments in negative.
This can cause problems if an actual keypoint is out of the image from the left or top
and actually has negative coordinates.
We currently assume that that can't happen.

After the user saves each image, two files will be saved along the image,
both with the same name but different extensions:

* .JSON with the keypoints, the measurements between them, the pixel size and the capture date. Its contents are a valid python dict.
* .JPG with the keypoints and the measurements between them painted on the image.
