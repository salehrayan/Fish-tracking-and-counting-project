import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import *
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch import optim
import torch
from torch import nn
from tqdm import tqdm

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

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.15, contrast=0.15)], p=0.5),
    transforms.RandomApply([transforms.RandomRotation(degrees=15)], p=0.5)
])
device = 'cpu'


class Dataset(Dataset):
    def __init__(self, input, labels):
        self.frames = input
        self.labels = labels

    def __len__(self):
        return self.frames.shape[0]

    def __getitem__(self, idx):
        frame_output = transform(
            torch.tensor(np.transpose(self.frames[idx, :, :, :], (2, 0, 1)), dtype=torch.float32).to(device))
        label_output = torch.tensor(self.labels[idx], dtype=torch.long).to(device)

        return label_output, frame_output


class ValidDataset(Dataset):
    def __init__(self, input, labels):
        self.frames = input
        self.labels = labels

    def __len__(self):
        return self.frames.shape[0]

    def __getitem__(self, idx):
        frame_output = torch.tensor(np.transpose(self.frames[idx, :, :, :], (2, 0, 1)), dtype=torch.float32).to(device)
        label_output = torch.tensor(self.labels[idx], dtype=torch.long).to(device)

        return label_output, frame_output


# dataset_train = Dataset(videos_with_rois_train, labels_train)
# dataloader_train = DataLoader(dataset_train, batch_size= 16, shuffle=True)

dataset_test = ValidDataset(videos_without_rois_test, labels_test)
dataloader_test = DataLoader(dataset_test, batch_size=16, shuffle=True)

model = nn.Sequential(
    # Conv layer 1:
    nn.Conv2d(3, 128, (3, 3)),
    nn.ReLU(),
    nn.BatchNorm2d(128),
    nn.MaxPool2d((2, 2)),

    # Conv layer 2:
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.BatchNorm2d(64),
    nn.MaxPool2d((2, 2)),

    # Conv layer 3:
    nn.Conv2d(64, 32, (3, 3)),
    nn.ReLU(),
    nn.BatchNorm2d(32),
    nn.MaxPool2d((2, 2)),

    nn.Conv2d(32, 16, (3, 3)),
    nn.ReLU(),
    nn.BatchNorm2d(16),
    nn.MaxPool2d((2, 2)),

    nn.Flatten(),

    # fully connected layers:
    nn.Linear(5888, 128),  # The dimensions here depend on the input size and the previous layers
    nn.ReLU(),
    nn.Linear(128, 9),

)
model.to(device)

optimizer = optim.Adam(params=model.parameters(), lr=0.0001, weight_decay=5e-4)
loss_fn = nn.CrossEntropyLoss()

checkpoint = torch.load('bestmodelcheckpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
model.to(device)

count = 0
with torch.no_grad():
    for t in range(0, videos_without_rois_test.shape[0]):
        frame = videos_without_rois_test[t, :, :, :]

        input = torch.tensor(np.transpose(videos_without_rois_test[t, :, :, :], (2, 0, 1)), dtype=torch.float).unsqueeze(
            0).to(device)
        output = model(input)
        count = count + torch.argmax(torch.nn.functional.softmax(output, dim=1).cpu().detach()).item()

        # im_with_keypoints[posy, :, 0] = 255

        plt.subplot(1, 1, 1)
        plt.title(f'Computer Count: {count}, Real Count: {np.sum(labels_test[0:t], dtype=np.int32)}')
        plt.imshow(frame)

        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
        plt.clf()
