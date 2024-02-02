import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import os
from utils import *
from sklearn.model_selection import train_test_split
import ast
from tqdm import tqdm
import time

video_paths = []
ROI_paths = []
label_paths = []

for root, directories, files in os.walk(r'dataset'):
    for file in files:
        file_path = os.path.join(root, file)

        if file.endswith('.avi'):
            video_paths = np.append(video_paths, file_path)

        elif file.endswith('ROI.txt'):
            ROI_paths = np.append(ROI_paths, file_path)

        elif file.endswith('manual.txt'):
            label_paths = np.append(label_paths, file_path)

video_paths_train, video_paths_test = train_test_split(video_paths, test_size=0.25, random_state=42)
ROI_paths_train, ROI_paths_test = train_test_split(ROI_paths, test_size=0.25, random_state=42)
label_paths_train, label_paths_test = train_test_split(label_paths, test_size=0.25, random_state=42)

# labels_train, videos_without_rois_train = concat_vid_rois_and_labels(video_paths_train, ROI_paths_train, label_paths_train)
labels_test, videos_without_rois_test = concat_vid_rois_and_labels(video_paths_test, ROI_paths_test, label_paths_test)


fgbg = cv2.bgsegm.createBackgroundSubtractorLSBP()

params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 0
params.maxThreshold = 255
params.filterByColor = True
params.blobColor = 255
params.filterByArea = True
params.minArea = 10
# params.maxArea = 1000
# params.filterByInertia = True
params.minInertiaRatio = 0.001

# params.filterByCircularity = True
params.minCircularity = 0.001
# params.filterByConvexity = True
params.minConvexity = 0.001
detector = cv2.SimpleBlobDetector_create(params)


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
plt.figure(figsize=(14,7))
for i in range(0,300):

    fgmask = fgbg.apply(videos_without_rois_test[i, :, :, :])

    # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_ERODE, kernel)

    keypoints = detector.detect(fgmask)

    im_with_keypoints = cv2.drawKeypoints(videos_without_rois_test[i, :, :, :], keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.subplot(1, 3, 1)
    plt.imshow(fgmask, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 3, 2)
    plt.imshow(im_with_keypoints)
    for kp in keypoints:
        x, y = kp.pt
        plt.scatter(x, y, c='red', s=1)
    plt.subplot(1, 3, 3)
    plt.imshow(videos_without_rois_test[i, :, :, :])
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)
    plt.clf()