# MASK-O-DETECT- A Covid-19 Face Mask Detector.

MASK-O-DETECT is a real time covid-19 Face Mask Detector.

## Table of Content

 - [Introduction](https://github.com/jiyauppal/face-mask-detector#Introduction)
 - [Technologies](https://github.com/jiyauppal/face-mask-detector#Technologies)
 - [Installation](https://github.com/jiyauppal/face-mask-detector#Installations)
 - [Usage](https://github.com/jiyauppal/face-mask-detector#Usage)
 - [For Contributing](https://github.com/jiyauppal/face-mask-detector#For-Contributing)

## Introduction
-A real-time detector to check weather a person in the frame is wearing a mask or
not.</br>
-We first detect the face of the person from the image or video file.</br>
-We then process the face and predict the results using a face mask detector.</br> 
-The face mask detector model used is based on TensorFlow.</br>

## Technologies Used
- Programming Langugage: Python,opencv-python(cv2): for facial recognition and image processing,numpy: For Mathematical Operations on array data structures, keras: for deep learning(CNN) and machine learning to detect faces.
- IDE, Code Editor and other Tools
- VS Code
- git
- GitHub
## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install tensorflow
pip install numpy
pip install cv2
```

## Usage

```python
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import datetime
```

## For Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

