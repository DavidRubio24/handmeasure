# Handmesure

Utilities to locate points of interest in images of scanned hands to mesure the hand.


This project aims to detect points in two different poses: opened and closed.
Each pose has its own points to locate: 23 for the opened hands and 15 for the closed ones.

As of right now (2023/4/11) the strategy is as follows:
1. Use MediaPipe Hands to detect hand keypoints.
2. Use those keypoints and the edges of the hand as a reference to find the landmarks.
3. Show a human the points, so she can modify them if they are wrong.

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


it will install numpy and opencv as dependencies, but pip's opencv doesn't use the QT backend.

Install opencv from conda:

```conda install opencv```

Remove pip's opencv:

```pip uninstall opencv-contrib-python```


### Run

From the conda prompt, activate the environment:

```conda activate p10```

Change to the directory where the project is located, for example:

```
D:
cd handmeasure
```

To calibrate the images:

```python calibrate.py folder/with/uncalibrated/images folder/to/save/calibrated/images folder/to/move/uncalibrated/images```

To measure the hands:

```python handmeasure.py path/to/folder/with/images```
