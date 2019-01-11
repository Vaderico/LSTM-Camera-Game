import torch
import torch.nn as nn
import torchvision
from torchvision import models

class LSTMController(nn.Module):
    def __init__(self, hidden_dim, intermediate):
        super(LSTMController, self).__init__()
        self._intermediate = intermediate
        self.hidden_dim = hidden_dim


        self.res18_model = models.resnet18(pretrained=True)
        self.res18_model = nn.Sequential(*list(self.res18_model.children())[:-2])

        res18_out_dim = 512 * 7 * 7
        self.middle_out_dim = 500

        self.middle = nn.Linear(res18_out_dim, self.middle_out_dim)
        self.lstm = nn.LSTM(self.middle_out_dim, hidden_dim)

        self.hiddenToVel = nn.Linear(hidden_dim, 2)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, X):
        X = X.float()
        X = torch.squeeze(X)

        with torch.no_grad():
            X = self.res18_model(X)

        X = self.middle(X.view(len(X), -1))

        X = X.reshape(len(X),1,-1)

        lstm_out, self.hidden = self.lstm(X, self.hidden)
        vel_space = self.hiddenToVel(lstm_out.view(len(X), -1))

        return vel_space

