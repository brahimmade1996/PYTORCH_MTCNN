#!/usr/bin/env python
# coding: utf-8

# #### 定义ONet

# In[7]:


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


# In[ ]:





# In[8]:


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias, 0.1)


# #### 定义ONet
# 
# 1. ONet和RNet有些像，只不过这一步还增加了landmark位置的回归
# 
# 2. 输入大小调整为48*48 ,RNet输出的Bound即offset

# In[9]:


#input 48*48*3
class ONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=2)
        self.out = nn.Linear(in_features=2*2*128,out_features=256)
        
        self.det = nn.Linear(in_features=256,out_features=1)
        self.bound = nn.Linear(in_features=256,out_features=4)
        self.landmark = nn.Linear(in_features=256,out_features=10)
        self.apply(weights_init)
        pass
    
    def forward(self,tensor):
        #layer input
        input = tensor
        #layer 1
        t=self.conv1(input)
        t=F.relu(t)
        #print('ONet conv1 shape:',t.shape)
        t=F.max_pool2d(t,kernel_size=3,stride=2)
        #print('ONet MP1:3*3 shape:',t.shape)
        #layer 2
        t=self.conv2(t)
        t=F.relu(t)
        #print('ONet conv2 shape:',t.shape)
        t=F.max_pool2d(t,kernel_size=3,stride=2)
        #print('ONet MP2:3*3 shape:',t.shape)
        #layer3
        t=self.conv3(t)
        t=F.relu(t)
        #print('ONet conv3 shape:',t.shape)
        t=F.max_pool2d(t,kernel_size=2,stride=2)
        #print('ONet MP2:2*2 shape:',t.shape)
        #layer 4
        t=self.conv4(t)
        t=F.relu(t)
        #print('ONet conv4 shape:',t.shape)
        t=t.reshape(-1,128*2*2)
        #layer out
        t=self.out(t)
        #print('ONet out shape:',t.shape)
        #out label face B*2
        det = self.det(t)
        label = F.relu(det)
        print('Rnet out label shape:',label.shape)
        #out bounding box (B*4)
        bound = self.bound(t)
        offset = F.relu(bound)
        print('Rnet out offset shape:',offset.shape)
        #landmark = self.landmark(t)
        landmark = self.landmark(t)
        landmark = F.relu(landmark)
        print('Rnet out offset shape:',landmark.shape)
        return label,offset,landmark
    pass

t=torch.randn([4,3,48,48])

print(t.shape)

onet=ONet()

label,offset,landmark = onet(t)


# In[ ]:




