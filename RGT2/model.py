import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import kaiming_uniform, xavier_uniform
from settings import *
import torch
from torch import nn, optim



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,4, kernel_size = (5,5), padding = (2, 2)),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(2))
        kaiming_uniform(self.layer1[0].weight)
        self.layer2 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size = (5,5), padding = (2,2)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        kaiming_uniform(self.layer2[0].weight)
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size = (5,5), padding = (2,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        kaiming_uniform(self.layer3[0].weight)
        self.fc1 = nn.Sequential(
            nn.Linear(4096, 4*n_bins),
            nn.Sigmoid()
        )
        xavier_uniform(self.fc1[0].weight)
        self.fc2 = nn.Sequential(
            nn.Linear(4*n_bins, 4*n_bins),
            nn.Sigmoid()
        )
        xavier_uniform(self.fc2[0].weight)
        self.out = nn.Softmax2d()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(-1, 4096)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, 4 , 1, 1025)
        x = self.out(x)
        return x
