import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DARQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DARQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)
        self.LSTM = nn.LSTM(input_size=512, 
                            hidden_size=2, 
                            num_layers=2, 
                            bias=True, 
                            batch_first=True, 
                            dropout=0.2)
        self.Transformer = nn.TransformerEncoderLayer(d_model=512, 
                                                nhead=2, 
                                                dim_feedforward=2048, 
                                                dropout=0.3, 
                                                activation="relu", 
                                                layer_norm_eps=1e-05, 
                                                batch_first=True, 
                                                norm_first=False, 
                                                device=None, 
                                                dtype=None)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x_new = x.view(x.size(0), -1).unsqueeze(0)
        # x = self.head(x.view(x.size(0), -1))
        h_0 = torch.randn(2, x_new.shape[0], 2)
        c_0 = torch.randn(2, x_new.shape[0], 2)
        # print('before: ', x_new.shape)
        # print('after: ', self.Transformer(x_new).shape)
        x_new = self.Transformer(x_new)
        output, (h_0, c_0) = self.LSTM(x_new, (h_0, c_0))
        output = output.squeeze(0)
        return output