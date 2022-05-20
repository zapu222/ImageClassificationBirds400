import os
import cv2
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from utils import augment_image


class Birds400(Dataset):
    def __init__(self, path, task="train", img_size=224, augment=False):  # path = .\\BIRDS_400
        self.task = task
        self.img_size = img_size
        self.augment = augment

        labels_file = pd.read_csv(os.path.join(path, "class_dict.csv"))

        self.labels = {}
        for _, row in labels_file.iterrows():
            self.labels[row['class']] = row['class_index']
        self.indices = {v: k for k, v in self.labels.items()}

        self.images = []
        if task == "train":
            folders = os.listdir(os.path.join(path, "train"))
        if task == "valid":
            folders = os.listdir(os.path.join(path, "valid"))
        if task == "test":
            folders = os.listdir(os.path.join(path, "test"))

        for folder in folders:
            for img in os.listdir(os.path.join(path, task, folder)):
                self.images.append([folder, os.path.join(path, task, folder, img)])

        random.shuffle(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx][1]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size),0,0, cv2.INTER_LINEAR)

        if self.augment:
            image = augment_image(image)
        
        transform = ToTensor()
        x = transform(image)

        label = self.images[idx][0]
        label = self.labels[label]

        y = np.zeros((1,len(self.labels)))
        y[0][label] = 1

        return x, y