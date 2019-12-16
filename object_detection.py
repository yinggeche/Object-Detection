import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters
import math
from sklearn.cluster import KMeans
from scipy.io import loadmat
train_path='../CarTrainImages'
test_path='../CarTestImages'
def readin(directory_name):
    array_of_img=[]
    for filename in os.listdir(r""+directory_name):
        img = cv2.imread(directory_name + "/" + filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        array_of_img.append(img)
    return array_of_img
def SIFTcorner(img,blur_times=0,draw=False,dist=0):
    sift = cv2.xfeatures2d.SIFT_create()
    for _ in range(blur_times):
        img= cv2.GaussianBlur(img, (5, 5), 1)
    keypoints = sift.detect(img, None)
    output = np.array([keypoint.pt for keypoint in keypoints])
    if draw:
      drawcorner(img,output)
    #print(len(output))
    return output
def drawcorner(img,corner):
    plt.ion()
    plt.imshow(img, cmap='gray')
    plt.plot([p[0] for p in corner],[p[1] for p in corner],'+')
    plt.ioff()
    plt.show()
def extractpatch(img,corner,patch_size=25,draw=False):
    train_patch=[]
    offset = int(patch_size / 2)
    for x,y in corner:
        x=int(x)
        y=int(y)
        #print(x,y)
        if x - offset >= 0 and x + offset < img.shape[1] and y - offset >= 0 and y + offset < img.shape[0]:
            train_patch.append(img[y - offset:y + offset + 1, x - offset:x + offset + 1])
    result=np.reshape(train_patch,(len(train_patch),patch_size**2))
    if draw:
        plt.ion()
        plt.imshow(img, cmap='gray')
        for x,y in corner:
            plt.gca().add_patch(plt.Rectangle((x-offset, y-offset),patch_size, patch_size, edgecolor='r', fill=False, linewidth=2))
        plt.ioff()
        plt.show()
    #print('corner',np.array(corner).shape)
    #print('hh',np.array(result).shape)
    return result
def clusterpatches(allpatch,clusternumber=21):
    cluster = KMeans(n_clusters=clusternumber, max_iter=2000)
    cluster.fit(allpatch)
    return cluster
def assignlabels(imgs,cluster,clusternumber=21,patch_size=25):
    offset = int(patch_size / 2)
    displacement_vectors = [[] for _ in range(clusternumber)]
    for img in imgs:
        corner=SIFTcorner(img)
        object_center = np.array([img.shape[1] / 2, img.shape[0] / 2])
        for x,y  in corner:
            x=int(x)
            y=int(y)
            if x - offset >= 0 and x + offset < img.shape[1] and y - offset >= 0 and y + offset < img.shape[0]:
                patch = img[y - offset:y + offset + 1, x - offset:x + offset + 1]
                patch = np.reshape(patch, (patch_size ** 2))
                label = cluster.predict([patch])[-1]
                displacement_vectors[label].append(np.array([x, y]) - object_center)
    #print(np.array(displacement_vectors[1]).shape)
    return displacement_vectors
def testimage(img,corner,cluster,disp_vectors,show_vote=False,show_threshold=False,show_box=True,threshold=0.92):
    vote_image=np.zeros(img.shape)
    patches=extractpatch(img,corner)
    i=0
    for patch in patches:
        label = cluster.predict([patch])[-1]
        for x,y in disp_vectors[label]:
            dis_x=int(corner[i][0]-x)
            dis_y=int(corner[i][1]-y)
            if 0 <= dis_x < img.shape[1] and 0 <= dis_y < img.shape[0]:
                vote_image[dis_y, dis_x] += 1
        i+=1
    vote_image= cv2.dilate(vote_image, np.ones((5, 5)), iterations=3)
    if show_vote:
        plt.ion()
        plt.imshow(vote_image, cmap='gray')
        plt.ioff()
        plt.show()
    left_top=[]
    center = np.argwhere(vote_image >= np.max(vote_image) * threshold)
    if show_threshold:
      plt.ion()
      plt.imshow(img, cmap='gray')
      plt.plot([p[1] for p in center], [p[0] for p in center], '+')
      plt.ioff()
      plt.show()
    for a in range(len(center)):
        is_center=True
        for b in range(len(center)):
          if a!=b:
            if abs(center[a][0]-center[b][0])<=20 or abs(center[a][1]-center[b][1])<=50:
                #print(center[a][1])
                if vote_image[center[a][0],center[a][1]]<=vote_image[center[b][0],center[b][1]]:
                     vote_image[center[a][0], center[a][1]]-=1
                     is_center=False
        if is_center:
            left_top.append([center[a][0]-20,center[a][1]-50])
    if show_box:
        plt.ion()
        plt.imshow(img, cmap='gray')
        plt.plot([p[1] for p in left_top], [p[0] for p in left_top], '+')
        for p in left_top:
          plt.gca().add_patch(plt.Rectangle((p[1],p[0]), 100, 40,edgecolor='b',fill=False,linewidth=2))
        plt.ioff()
        plt.show()
    return left_top
def cal_accuracy(left_top_array,showground=False):
    ground_truth = loadmat("../GroundTruth/CarsGroundTruthBoundingBoxes.mat")
    ground_truth = ground_truth["groundtruth"][0]
    acc=0
    for i in range(len(left_top_array)):
        temp_acc=0
        for j in range(len(ground_truth['topLeftLocs'][i])):
            #print("true")
            #print(ground_truth['topLeftLocs'][i][j])
            if len(left_top_array[i])>j:
                 #print("pre:")
                 #print(left_top_array[i][j])
                 #???????????????????????????
                 disp_x=100-abs(left_top_array[i][j][1]-ground_truth['topLeftLocs'][i][j][0]+1)
                 disp_y=40-abs(left_top_array[i][j][0]-ground_truth['topLeftLocs'][i][j][1]+1)
                 if disp_x>0 and disp_y>0:
                   area=(disp_x*disp_y)/4000
                   if area>=0.5:
                       temp_acc+=1
        temp_acc=temp_acc/len(ground_truth['topLeftLocs'][i])
        #print(temp_acc)
        acc+=temp_acc
    acc=acc/100
    return acc
def main():
#train part
    print("------train begin------")
    print("read in images")
    train_img=readin(train_path)
    print("extract patches")
    allpatch=[]
    for img in train_img:
      corner=SIFTcorner(img,blur_times=1)
      allpatch+=(p for p in extractpatch(img,corner))
    print("cluster patches")
    cluster=clusterpatches(allpatch)
    print("assign labels")
    disp_vectors=assignlabels(train_img,cluster)
    print("------train finish------")
    print("------test begin-------")
    test_img=readin(test_path)
    left_top=[[] for _ in range(len(test_img))]
    i=0
    for img1 in test_img:
        corner=SIFTcorner(img1,blur_times=1,draw=False)
        left_top[i]=testimage(img1,corner,cluster,disp_vectors)
        i+=1
    #print(np.array(left_top).shape)
    acc=cal_accuracy(left_top)
    print("accuracy is: ",acc)
main()
