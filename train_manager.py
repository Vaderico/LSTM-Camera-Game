import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms

import numpy as np
import cv2
import os
import random
import time
import copy

from dataset import SequenceDataset
from model import LSTMController

class TrainManager():
    def __init__(self, training_dir, intermediate, split, epochs, batch_size):
        self._training_dir = training_dir
        self._intermediate = intermediate
        self._split = split
        self._epochs = epochs
        self._batch_size = batch_size

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True

        self._model = LSTMController(2000, self._intermediate)
        self._model = self._model.to(self._device)

        self._dataloaders = {}
        self._dataset_sizes = {}
        self._partition_names = ['train', 'val']

    def load_data(self):
        # Shuffle and split the sequences into train, validation, and test set
        sequence_dirs = os.listdir(self._training_dir)
        if 'temp' in sequence_dirs:
            sequence_dirs.remove('temp')

        random.shuffle(sequence_dirs)
        threshold = int(len(sequence_dirs) * self._split)
        partition = {'train': sequence_dirs[threshold:],
                     'val': sequence_dirs[:threshold]}

        mean, std = self.mean_and_std();
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        image_datasets = {x: SequenceDataset(partition[x], self._training_dir, data_transforms) 
                for x in self._partition_names}

        self._dataloaders = {x: DataLoader(image_datasets[x], batch_size=1, shuffle=True,
                                num_workers=1) for x in self._partition_names}

        self._dataset_sizes = {x: len(image_datasets[x]) for x in self._partition_names}

    def mean_and_std(self):
        print("Calculating mean and std dev")
        dataset_mean = [0.0, 0.0, 0.0]
        dataset_std = [0.0, 0.0, 0.0]
        count = 0

        for sequence in os.listdir(self._training_dir):
            seq_path = os.path.join(self._training_dir, sequence)

            for file_name in os.listdir(seq_path):
                if not file_name.endswith('.png'):
                    continue;
                count += 1

                image_path = os.path.join(seq_path, file_name)
                img = cv2.imread(image_path)
                mean, std = cv2.meanStdDev(img)

                dataset_mean += (np.transpose(mean)[0] / 255)
                dataset_std += (np.transpose(std)[0] / 255)
                
        return dataset_mean / count, dataset_std / count
        
    def train_model(self):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self._model.parameters(), lr=0.1)
        # Decay LR by a factor of 0.1 every 7 epochs
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        since = time.time()

        best_model_wts = copy.deepcopy(self._model.state_dict())
        best_loss = float('inf')

        for epoch in range(self._epochs):
            print('Epoch {}/{}'.format(epoch, self._epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in self._partition_names:
                if phase == 'train':
                    scheduler.step()
                    self._model.train()  # Set model to training mode
                else:
                    self._model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0


                # Iterate over data.
                for X, y in self._dataloaders[phase]:
                    X = X.to(self._device)
                    y = y.to(self._device)
                    print('Loading sequence===============')

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self._model(X)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, torch.squeeze(y))

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward(retain_graph=True)
                            optimizer.step()
                    
                    # statistics
                    running_loss += loss.item() * X.size(0)

                epoch_loss = running_loss / self._dataset_sizes[phase]
                print('{} Loss: {:.4f}'.format( phase, epoch_loss)) 

                # deep copy the model
                if phase == 'val' and epoch_loss > best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(self._model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

