# This is a harris corner detection and matching tools
import sys
import argparse
import math
import numpy as np
import cv2
import os
from sklearn.cluster import KMeans

def harris_corner(filename):
    """ Find interesting corner features based on Harris Corner Detection
    """
    window_size = 3
    r = window_size//2
    img = cv2.imread(filename)
    img1 = img
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Convert image into gray scale
    gray = np.float32(gray)
    # Convert image into float32
    dst = cv2.cornerHarris(gray,2,3,0.04)
    # Result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    # print(dst.shape)
    img[dst>0.01*dst.max()]=[0,0,255]
    # posc,posr = dst>0.01*dst.max()
    cv2.imwrite('harris.png',img)
    # Store the harris corner coordinates
    good = np.argwhere(dst > 0.01*dst.max())
    # print(good.shape)
    # print(good)
    # px = gray[4-r:4+r+1,5-r:5+r+1]
    # # [4-r:4+r, 5-r:5+r]
    # print(px)
    patches = np.array([])
    for item in good:
        x = item[1]
        y = item[0]
        pts1 = (x-r,y-r)
        pts2 = (x+r,y+r)
        cv2.rectangle(img1, pts1, pts2, (255,0,0), 1)
        # patch = gray[x-r:x+r+1,y-r:y+r+1]
        # img1[x-r:x+r+1,y-r:y+r+1] = [0,0,255]
        # patch_re = patch.reshape(-1)
        # patches = np.append(patches,patch_re)
    # # print(res)
    cv2.imwrite('patches.png',img1)
    # return patches
    # max = 0.01 * dst.max()
    # good = []
    # for i, item in enumerate(dst):
    #     if item > 0.01 * dst.max():
    #         good.append(item)
    # print(good)

def kmeans_cluster(k, X):
    y_pred = KMeans(n_clusters=k, random_state=9).fit_predict(X)
    return y_pred


if __name__ == "__main__":
    path = '../CarTrainImages/'
    path_1 = '../CarTrainImages/train_car001.jpg'
    # path_2 = '../CarTrainImages/train_car002.jpg'
    # patch_1 = harris_corner(path_1)
    # patch_2 = harris_corner(path_2)
    # patches = np.array([])
    # patches = patches + patch_1
    # patches = patches + patch_2
    harris_corner(path_1)
    # files = os.listdir(path)
    # patches = []
    # for file in files:
    #     # print(file)
    #     patch = harris_corner(path+file)
    #     patches = np.append(patches,patch)
    # # print(res)
    # patches = patches.reshape(1,-1)
    # patches = patches.reshape(1,-1)
    # print(patches)
    # k = 20
    # cluster_res = kmeans_cluster(k, patches)
