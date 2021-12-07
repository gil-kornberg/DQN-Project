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

class DAQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DAQN, self).__init__()
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
        self.attention = torch.nn.MultiheadAttention(embed_dim=128, 
                                                     num_heads=8, 
                                                     dropout=0.0, 
                                                     bias=True, 
                                                     add_bias_kv=False, 
                                                     add_zero_attn=False, 
                                                     kdim=None, 
                                                     vdim=None, 
                                                     batch_first=True, 
                                                     device=None, 
                                                     dtype=None)
        self.key = nn.Linear(512, 128)
        self.query = nn.Linear(512, 128)
        self.value = nn.Linear(512, 128)
        self.head2 = nn.Linear(128, 2)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # print('before reshape: ', x.shape)
        # print(x.view(x.size(0), -1).shape)
        x_att = x.view(x.size(0), -1).unsqueeze(0)
        key = F.relu(self.key(x_att))
        query = F.relu(self.query(x_att))
        value = F.relu(self.value(x_att))
        x_new, x_new_weights = self.attention(key, query, value)
        x_new = self.head2(F.relu(x_new.squeeze(0)))
        # print(x_new.shape, x_new_weights.shape, self.head2(x_new).shape)
        # x = self.head(x.view(x.size(0), -1))
        return x_new
