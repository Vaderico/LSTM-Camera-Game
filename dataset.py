import torch
import torchvision
from torch.utils import data

import numpy as np
from PIL import Image
import os
import csv
import sys
import cv2

class SequenceDataset(data.Dataset):
    def __init__(self, sequence_dirs, training_dir, transform):
        self._sequence_dirs = sequence_dirs
        self._training_dir = training_dir
        self._transform = transform

    def __len__(self):
        return len(self._sequence_dirs)

    def __getitem__(self, index):
        # Select sequence
        seq_dir = self._sequence_dirs[index]
        seq_path = os.path.join(self._training_dir, seq_dir)

        # Create empty array
        # images = np.array([], dtype=np.float64).reshape(0,3,224,224)
        images_list = []
        list_dir = os.listdir(seq_path)
        list_dir.sort()
        # print(list_dir)

        for file_name in list_dir:
            if not file_name.endswith('.png'):
                continue;
            image_path = os.path.join(seq_path, file_name)
            img = np.array(Image.open(image_path))
            # img = np.transpose(img, (2,0,1))
            # img = img.reshape((224,224, 3))
            img = self._transform(img)
            # images = np.concatenate((images, img), axis=0)
            images_list.append(img)

        # Load data and get label
        X = torch.stack(images_list)
        # print(X.shape)

        velocities_path = os.path.join(self._training_dir, seq_dir, 'velocities.csv')
        with open(velocities_path, newline='') as csvfile:
            data = [list(map(float, row)) for row in csv.reader(csvfile)]
        y = torch.tensor(data)

        return X, y

if __name__ == "__main__":
    import random
    from matplotlib import pyplot as plt
    from torchvision import datasets, models, transforms

    training_dir = "training_data/up_down"
    sequence_dirs = os.listdir(training_dir)
    if 'temp' in sequence_dirs:
        sequence_dirs.remove('temp')
    if 'pt' in sequence_dirs:
        sequence_dirs.remove('pt')

    # random.shuffle(sequence_dirs)
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    d = SequenceDataset(sequence_dirs, training_dir, data_transforms)

    # cv2.namedWindow("img",0)
    while True:
        for X, y in d:
            for i in range(len(X)):
                img = X[i,:,:,:].numpy().transpose(1,2,0).copy()
                img *= 50
                img += 127
                img = img.astype("uint8")
                # print(img.shape)
                # print(img)
                h,w,ch = img.shape

                plt.cla()
                plt.imshow(img)
                plt.title(str(y[i]))
                plt.pause(1)
                # cv2.imshow("img",img)
                # cv2.waitKey(300)

