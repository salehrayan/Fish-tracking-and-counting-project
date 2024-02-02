import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

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

video_paths_train, video_paths_test = train_test_split(video_paths, test_size=0.3, random_state=42)
ROI_paths_train, ROI_paths_test = train_test_split(ROI_paths, test_size=0.3, random_state=42)
label_paths_train, label_paths_test = train_test_split(label_paths, test_size=0.3, random_state=42)

sample_labels, sample_videoes_without_roi, posy = load_video_with_ROI_with_separate_label(video_paths_train[6],
                                                                                          ROI_paths_train[6],
                                                                                          label_paths_train[6])

posy = math.floor(posy)

labels_train, videos_without_rois_train = concat_vid_rois_and_labels(video_paths_train, ROI_paths_train,
                                                                     label_paths_train)
# labels_test, videos_without_rois_test = concat_vid_rois_and_labels(video_paths_test, ROI_paths_test, label_paths_test)


fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 0
params.maxThreshold = 100
params.filterByColor = True
params.blobColor = 255
params.filterByArea = True
params.minArea = 10
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
required_appearances = 3

random_forest = RandomForestClassifier()
decision_tree = DecisionTreeClassifier()
fgmasks = []
plt.figure(figsize=(14, 7))
for t in range(0, sample_videoes_without_roi.shape[0]):
    img = sample_videoes_without_roi[t, :, :, :]
    # define each block as 4x4 cells of 64x64 pixels each
    cell_size = (32, 32)  # h x w in pixels
    block_size = (2, 2)  # h x w in cells
    win_size = (8, 6)  # h x w in cells

    nbins = 9  # number of orientation bins
    img_size = img.shape[:2]  # h x w in pixels

    # create a HOG object
    hog = cv2.HOGDescriptor(
        _winSize=(win_size[1] * cell_size[1],
                  win_size[0] * cell_size[0]),
        _blockSize=(block_size[1] * cell_size[1],
                    block_size[0] * cell_size[0]),
        _blockStride=(cell_size[1], cell_size[0]),
        _cellSize=(cell_size[1], cell_size[0]),
        _nbins=nbins
    )
    n_cells = (img_size[0] // cell_size[0], img_size[1] // cell_size[1])

    # find features as a 1xN vector, then reshape into spatial hierarchy
    hog_feats = hog.compute(img)
    fgmasks.append(hog_feats)
    # hog_feats = hog_feats.reshape(
    #     n_cells[1] - win_size[1] + 1,
    #     n_cells[0] - win_size[0] + 1,
    #     win_size[1] - block_size[1] + 1,
    #     win_size[0] - block_size[0] + 1,
    #     block_size[1],
    #     block_size[0],
    #     nbins)

    # break
# print(hog_feats.shape)

fgmasks = np.stack(fgmasks, axis=0)
# fgmasks = fgmasks.reshape((-1, 100 * 200))

RandomForestClassifier.fit(fgmasks, sample_labels)
predict = RandomForestClassifier.apply(fgmasks)

print(f'Accuracy Training: {np.mean(predict == sample_labels)}')

# keypoints = detector.detect(fgmask)
#
# im_with_keypoints = cv2.drawKeypoints(sample_videoes_without_roi[t, :, :, :], keypoints, np.array([]), (0, 0, 255),
#                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# keypoints = list(keypoints)

# im_with_keypoints[posy, :, 0] = 255
#
# plt.subplot(1, 3, 1)
# plt.imshow(fgmask, cmap='gray', vmin=0, vmax=255)
# plt.subplot(1, 3, 2)
# plt.title(f'Computer Count: {count}, Real Count: {np.sum(sample_labels[0:t], dtype=np.int32)}')
# plt.imshow(im_with_keypoints)
# for kp in keypoints:
#     x, y = kp.pt
#     plt.scatter(x, y, c='red', s=1)
# plt.subplot(1, 3, 3)
# plt.imshow(sample_videoes_without_roi[t, :, :, :])
# plt.tight_layout()
# plt.draw()
# plt.pause(1)
# plt.clf()
