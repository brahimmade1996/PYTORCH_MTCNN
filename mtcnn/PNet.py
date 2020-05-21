#!/usr/bin/env python
# coding: utf-8

# ##### 定义PNet

# In[6]:


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


# ###### 对权重进行初始化，使用正态分布

# In[7]:


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias, 0.1)


# ##### 定义PNet网络

# In[8]:



#input 12*12*3
class PNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=10,kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=10,out_channels=16,kernel_size=3)
        self.out = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3)
        
        self.det = nn.Conv2d(in_channels=32,out_channels=1,kernel_size=1)
        self.bound = nn.Conv2d(in_channels=32,out_channels=4,kernel_size=1)
        self.landmark = nn.Conv2d(in_channels=32,out_channels=10,kernel_size=1)
        self.apply(weights_init)
        pass
    
    def forward(self,tensor):
        #layer input
        input=tensor
        #layer 1
        t=self.conv1(input)
        t = F.relu(t)
        #print('pnet conv1 shape:',t.shape)
        t=F.max_pool2d(t,kernel_size=2,stride=2)
        #print('pnet mp1 shape:',t.shape)
        #layer 2
        t=self.conv2(t)
        t = F.relu(t)
        #print('pnet conv2 shape:',t.shape)
        #layer 3
        t = self.out(t)
        #print('pnet out shape:',t.shape)
        # t = F.relu(t)
        #out label face 1*1*2
        det = self.det(t)
        label = torch.sigmoid(det)
        #out bounding box (1*1*4)
        bound = self.bound(t)
        offset = F.relu(bound)
        #landmark = self.landmark(t)
        return label,offset
    pass


# ##### 测试PNet

# In[9]:


t = torch.rand([4,3,12,12])

label = torch.randn([4])

print(label)

print(t.shape)

pnet =PNet()

plabel,offset = pnet(t)

plabel = plabel.squeeze()

mask = torch.ge(plabel,0)

valid_gt_cls = torch.masked_select(plabel,mask)

prob_ones = torch.ge(valid_gt_cls,0.2)

print('b',plabel)

print('a:',mask)

print('c',valid_gt_cls)

print('c',prob_ones)


# In[ ]:




