#!/usr/bin/env python
# coding: utf-8

# #### 计算损失函数

# In[ ]:


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


# In[ ]:





# In[ ]:


class LossFn:
    def __init__(self, cls_factor=1, box_factor=1, landmark_factor=1):
        # loss function
        self.cls_factor = cls_factor
        self.box_factor = box_factor
        self.land_factor = landmark_factor
        self.loss_cls = nn.BCELoss() 
        # binary cross entropy
        self.loss_box = nn.MSELoss() 
        # mean square error
        self.loss_landmark = nn.MSELoss()


    def cls_loss(self,gt_label,pred_label):
        pred_label = torch.squeeze(pred_label)
        gt_label = torch.squeeze(gt_label)
        # get the mask element which >= 0, only 0 and 1 can effect the detection loss
        mask = torch.ge(gt_label,0)
        valid_gt_label = torch.masked_select(gt_label,mask)
        valid_pred_label = torch.masked_select(pred_label,mask)
        return self.loss_cls(valid_pred_label,valid_gt_label)*self.cls_factor


    def box_loss(self,gt_label,gt_offset,pred_offset):
        pred_offset = torch.squeeze(pred_offset)
        gt_offset = torch.squeeze(gt_offset)
        gt_label = torch.squeeze(gt_label)

        #get the mask element which != 0
        unmask = torch.eq(gt_label,0)
        mask = torch.eq(unmask,0)
        #convert mask to dim index
        chose_index = torch.nonzero(mask.data)
        chose_index = torch.squeeze(chose_index)
        #only valid element can effect the loss
        valid_gt_offset = gt_offset[chose_index,:]
        valid_pred_offset = pred_offset[chose_index,:]
        return self.loss_box(valid_pred_offset,valid_gt_offset)*self.box_factor

