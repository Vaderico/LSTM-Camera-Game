import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random
import time
import copy

from dataset import SequenceDataset
from model import LSTMController
from utils import memReport, cpuStats

class LSTMStepLearner(nn.Module):
    def __init__(self, input_dim, hidden_dim, sequence_size):
        super(LSTMStepLearner, self).__init__()

        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.hiddenToVel = nn.Linear(hidden_dim, 2)
        self.init_hidden()
    
    def init_hidden(self):
        self.hidden = (torch.zeros(1, 1, self.hidden_dim).cuda(),
                       torch.zeros(1, 1, self.hidden_dim).cuda())
    
    def forward(self, X):
        X = X.reshape(len(X),1,-1)
        lstm_out, self.hidden = self.lstm(X, self.hidden)
        vel_space = self.hiddenToVel(lstm_out.view(len(X), -1))
        return vel_space

class TrainManager():
    def __init__(self, training_dir=None, models_dir=None, split=None, epochs=None, batch_size=None):
        if training_dir is None:
            pass

        self._training_dir = training_dir
        self._models_dir = models_dir
        self._split = split
        self._epochs = epochs
        self._batch_size = batch_size

        torch.backends.cudnn.benchmark = True

        self._model = LSTMController().cuda()

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

        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])

        image_datasets = {x: SequenceDataset(partition[x], self._training_dir, data_transforms) 
                for x in self._partition_names}

        self._dataloaders = {x: DataLoader(image_datasets[x], batch_size=1, shuffle=True,
                                num_workers=1) for x in self._partition_names}

        self._dataset_sizes = {x: len(image_datasets[x]) for x in self._partition_names}

    def mean_and_std(self):
        print("calculating mean and std dev...")
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
        self._criterion = nn.MSELoss()
        optimizer = optim.Adam(self._model.parameters(), lr=0.0001)

        since = time.time()
        best_loss = float('inf')

        train_loss_list = []
        val_loss_list = []

        self._seq_len = 100

        for epoch in range(self._epochs):
            # Set model to training mode
            self._model.train()  

            # Iterate over data.
            for i, (X, y) in enumerate(self._dataloaders['train']):
                # X = torch.randn(self._seq_len, 20).cuda()
                # y = torch.tensor(self.generate_step_tensor(self._seq_len, 10)).float().cuda()

                optimizer.zero_grad()
                self._model.init_hidden()
                            
                X = torch.squeeze(X.cuda())
                predict = self._model(X)

                loss = self._criterion(predict, y.cuda())
                self._model.train()  
                loss.backward(retain_graph=True)
                optimizer.step()

                print("[Epoch: %d/%d, Sample: %d] loss: %0.4f " % (
                    epoch + 1, self._epochs, i, loss.item()), end='\r', flush=True)

            # print training and validation loss
            train_loss = self.test('train')
            val_loss = self.test('val')
            print('Epoch: %d, train loss: %0.4f, validation loss: %0.4f' % (epoch + 1, train_loss, val_loss))
                   

            # plot training and validation loss
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            self.plot_loss(train_loss_list, val_loss_list)

            # save this model if it has the lowest validation loss
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self._model, os.path.join(self._models_dir, "model.pt"))

        # print total time taken to train
        time_elapsed = time.time() - since
        print('Training complete in %0.0fm %0.0fs' % (time_elapsed / 60, time_elapsed % 60))

    def test(self, phase):
        loss = 0.0
        with torch.no_grad():
            for X, y in self._dataloaders[phase]:
            # for i in range(200):
                # X = torch.randn(self._seq_len, 20).cuda()
                # y = torch.tensor(self.generate_step_tensor(self._seq_len, 10)).float().cuda()

                X = torch.squeeze(X).cuda()
                self._model.init_hidden()
                predict = self._model(X)
                loss += self._criterion(predict, y.cuda()).item()

        return loss / self._dataset_sizes[phase]


    def plot_loss(self, train_loss_list, val_loss_list):
        plt.cla()
        plt.plot(train_loss_list, color="blue", label="Training Loss")
        plt.plot(val_loss_list, color="green", dashes=[6,2], label="Validation Loss")
        plt.legend(loc="best")
        plt.title("Training and Validation Loss vs Epochs")
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.draw()
        plt.pause(0.1)
        plt.savefig(os.path.join(self._models_dir, "plot"))

    def generate_step_tensor(self, seq_length, step_size):
        step_tensor = np.array([], dtype=np.float64).reshape(0, 2)
        small = True
        for i in range(seq_length):
            if i % step_size is 0:
                small = not small
            if small:
                step_tensor = np.concatenate((step_tensor, np.array([[6, 0]])))
            else:
                step_tensor = np.concatenate((step_tensor, np.array([[0, 6]])))
        return step_tensor

# Main function
if __name__ == "__main__" :
    trainer = TrainManager()
    trainer.train_model()

