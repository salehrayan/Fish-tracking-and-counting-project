import math

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

sample_labels, sample_videoes_without_roi, posy = load_video_with_ROI_with_separate_label(video_paths_train[6],
                                                                                    ROI_paths_train[6],
                                                                                    label_paths_train[6])

posy = math.floor(posy)

# labels_train, videos_without_rois_train = concat_vid_rois_and_labels(video_paths_train, ROI_paths_train, label_paths_train)
# labels_test, videos_without_rois_test = concat_vid_rois_and_labels(video_paths_test, ROI_paths_test, label_paths_test)



fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 0
params.maxThreshold = 100
params.filterByColor = True
params.blobColor = 255
params.filterByArea = False
params.minArea = 15
# params.maxArea = 1000
params.filterByInertia = False
params.minInertiaRatio = 0.001
params.filterByCircularity = False
params.minCircularity = 0.001
params.filterByConvexity = False
params.minConvexity = 0.001

detector = cv2.SimpleBlobDetector_create(params)


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


fishes = []
counted_fish = []
candidates = []
IDs = []
diameters = []
count = 0
required_appearances = 10

plt.figure(figsize=(14, 7))
for t in range(0, sample_videoes_without_roi.shape[0]):

    if t == 143:
        ad = 3
    fgmask = fgbg.apply(sample_videoes_without_roi[t, :, :, :])

    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    keypoints = detector.detect(fgmask)

    im_with_keypoints = cv2.drawKeypoints(sample_videoes_without_roi[t, :, :, :], keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    keypoints = list(keypoints)

    keypoints, fishes, candidates = tracking(keypoints, fishes, candidates, posy)
    keypoints, candidates, IDs = create_candidates(keypoints, candidates, IDs)
    fishes, candidates = promote_candidates_to_fish(fishes, candidates, required_appearances, IDs)
    count, counted_fish = counting(fishes, posy, count, counted_fish)

    im_with_keypoints[posy, :, 0] = 255

    plt.subplot(1, 3, 1)
    plt.imshow(fgmask, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 3, 2)
    plt.title(f'Computer Count: {count}, Real Count: {np.sum(sample_labels[0:t], dtype=np.int32)}')
    plt.imshow(im_with_keypoints)
    for kp in keypoints:
        x, y = kp.pt
        plt.scatter(x, y, c='red', s=1)
    plt.subplot(1, 3, 3)
    plt.imshow(sample_videoes_without_roi[t, :, :, :])
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)
    plt.clf()
