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

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
plt.figure(figsize=(12,7))
for i in range(300):
    # applying on each frame
    fgmask = fgbg.apply(videos_without_rois_test[i, :, :, :])

    # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_ERODE, kernel)

    plt.subplot(1, 2, 1)
    plt.imshow(fgmask, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(videos_without_rois_test[i, :, :, :])

    plt.tight_layout()
    plt.draw()
    plt.pause(0.2)
    plt.clf()
