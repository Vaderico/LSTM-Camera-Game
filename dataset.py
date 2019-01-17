import torch
import torchvision
from torch.utils import data

import numpy as np
from PIL import Image
import os
import csv

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
        images = np.array([], dtype=np.float64).reshape(0,3,224,224)

        for file_name in os.listdir(seq_path):
            if not file_name.endswith('.png'):
                continue;
            image_path = os.path.join(seq_path, file_name)
            img = np.array(Image.open(image_path))
            img = self._transform(img)
            img = np.transpose(img, (2,0,1))
            img = img.reshape((1,3,224,224))
            images = np.concatenate((images, img), axis=0)

        # Load data and get label
        X = torch.tensor(images)

        velocities_path = os.path.join(self._training_dir, seq_dir, 'velocities.csv')
        with open(velocities_path, newline='') as csvfile:
            data = [list(map(float, row)) for row in csv.reader(csvfile)]
        y = torch.tensor(data)

        return X, y

