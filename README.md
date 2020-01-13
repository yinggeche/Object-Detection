# Object-Detection
In this project, there are two parts to implement the Object Detection algorithm. 

# Keyword
1.  Generalized Hough Transforma (GHT)
2.  Visual Vocabulary
3.  Euclidean distance (SSD)

## Algorithm
### Training Stage
1. Given a set of training cropped images prepare a vocabulary and GHT translation vectors.
2. Use a Harris Corner detector to collect intereseting points.
3. At each interesting point, extract a fixed size image patch and use the vector of raw pixel intensities as the descriptor.
4.  Cluster the patches by K-means to constitute a "visual vocabulary".
5.  Go back to training images and assign their patches to visual words in vocabulary by using the closest Euclidean distance.
6.  Record the possible displacement vectors between visual word and object center.
### Testing Stage
1.  Detect interesting points by corner detection.
2.  Use a fixed image patch to create a raw pixel descriptor.
3.  Assign to each patch a visual word.
4.  Let visual word occurrence vote for the position of the object using the stored displacement vectors.
5.  After all votes are cast, analyze the votes and predict where the object occurs.
6.  Compute the accuracy of the predictions.

## Data
### Train Images
550 cropped training images of cars, eaach 40 * 100 pixels
### Test Images
100 test images
### Ground Truth
A list of locations: topLeftLocs = [x1, y1; x2, y2; ...; xn, yn]

## Packages
```python
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters
import math
from sklearn.cluster import KMeans
from scipy.io import loadmat
```
