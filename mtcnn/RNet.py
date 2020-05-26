#!/usr/bin/env python
# coding: utf-8

# ##### 定义RNet

# In[1]:


#torch package
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import sys
sys.path.append('../')

# add other package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tool.plotcm import plot_confusion_matrix
import tool.image_tools

import pdb

from collections import OrderedDict
from collections import namedtuple
from itertools import product

#torch.set_printoptions(linewidth=120)


# In[2]:


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias, 0.1)


# In[3]:


#input 24*24*3
class RNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=28,kernel_size=3,  stride=1)
        self.conv2 = nn.Conv2d(in_channels=28,out_channels=48,kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=48,out_channels=64,kernel_size=2, stride=1)
        self.out = nn.Linear(in_features=3*3*64,out_features=128)
        
        self.det = nn.Linear(in_features=128,out_features=1)
        self.bound = nn.Linear(in_features=128,out_features=4)
        self.landmark = nn.Linear(in_features=128,out_features=10)
        self.apply(weights_init)
        pass
    
    def forward(self,tensor):
        #layer input
        input = tensor
        #layer 1
        t=self.conv1(input)
        t=F.relu(t)
        t=F.max_pool2d(t,kernel_size=2,stride=2)
        #print('RNet conv1 shape:',t.shape)
        #print('RNet MP1 3*3 shape:',t.shape)
        #layer 2
        t=self.conv2(t)
        t=F.relu(t)
        t=F.max_pool2d(t,kernel_size=2,stride=2)
        #print('RNet MP2 3*3 shape: ',t.shape)
        #layer3
        t=self.conv3(t)
        t=F.relu(t)
        #print('RNet conv3 shape: ',t.shape)
        #layer out
        t = t.reshape(-1, 3*3*64)
        #print('RNet out reshape: ',t.shape)
        t=self.out(t)
        #print('RNet out shape: ',t.shape)
        #out label face B*2
        det = self.det(t)
        label = F.relu(det)
        #print('Rnet out label shape:',label.shape)
        #out bounding box (B*4)
        bound = self.bound(t)
        offset = F.relu(bound)
        #print('Rnet out offset shape:',offset.shape)
        #out landmark 
        #landmark = self.landmark(t)
        #landmark = F.relu(landmark)
        #print('Rnet out offset shape:',landmark.shape)
        return label,offset
    pass

t=torch.randn([4,3,24,24])



label=torch.randn([4])

rnet=RNet()

rlabel,roffset = rnet(t)


#cm = confusion_matrix(label, rlabel.argmax(dim=1),labels=[1])


print('a',rlabel.shape)

