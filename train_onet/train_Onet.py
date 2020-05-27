#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[4]:


#导入公共文件
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import sys
sys.path.append('../')

# add other package
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from tool.plotcm import plot_confusion_matrix

import pdb

from collections import OrderedDict

from collections import namedtuple

from itertools import product

#torch.set_printoptions(linewidth=120)

from mtcnn.ONet import ONet

from mtcnn.mtcnn import RunBuilder

from mtcnn.LossFn import LossFn

from tool.imagedb import ImageDB

from tool.imagedb import TrainImageReader

from tool import image_tools

import datetime

torch.set_grad_enabled(True)


# In[ ]:





# In[6]:


annotation_file = "../image/48/imglist_anno_48.txt"
model_store_path = "../model"

params = OrderedDict(
    lr = [.01]
    ,batch_size = [2000]
    ,device = ["cpu"]
    ,shuffle = [True]
)

end_epoch = 10

frequent = 10


# In[ ]:





# In[9]:


def train_onet(imdb=None):
    
    if imdb == None:
        imagedb = ImageDB(annotation_file)
        imdb = imagedb.load_imdb()
        #print(imdb.num_images)
        imdb = imagedb.append_flipped_images(imdb)
        
    for run in RunBuilder.get_runs(params):
        use_cuda= True if run.device == 'cuda' else False
        #create model path
        if not os.path.exists(model_store_path):
            os.makedirs(model_store_path)
        
        lossfn = LossFn()
        
        network = ONet(is_train=True, use_cuda=use_cuda)
        
        if use_cuda:
            network.cuda()
        
        optimizer = torch.optim.Adam(network.parameters(), lr=run.lr)
        
        train_data=TrainImageReader(imdb,24,run.batch_size,shuffle=True)
        
        comment = f'-{run}'
        
        for epoch in range(end_epoch):
            
            train_data.reset() # shuffle
            
            epoch_acc = 0.0
            
            for batch_idx,(image,(gt_label,gt_bbox,gt_landmark))in enumerate(train_data):
                
                im_tensor = [ image_tools.convert_image_to_tensor(image[i,:,:,:]) for i in range(image.shape[0]) ]
                
                im_tensor = torch.stack(im_tensor)

                im_tensor = Variable(im_tensor)
                
                gt_label = Variable(torch.from_numpy(gt_label).float())

                gt_bbox = Variable(torch.from_numpy(gt_bbox).float())
                
                gt_landmark = Variable(torch.from_numpy(gt_landmark).float())
                
                cls_pred, box_offset_pred,landmark_offset_pred = network(im_tensor)
                
                cls_loss = lossfn.cls_loss(gt_label,cls_pred)
                
                box_offset_loss = lossfn.box_loss(gt_label,gt_bbox,box_offset_pred)
                
                landmark_loss = lossfn.landmark_loss(gt_label,gt_landmark,landmark_offset_pred)
                
                all_loss  = cls_loss*1.0+box_offset_loss*0.5
                
                cls_pred, box_offset_pred = network(im_tensor)
                
                if batch_idx%frequent==0:
                    accuracy=compute_accuracy(cls_pred,gt_label)
                    accuracy=compute_accuracy(cls_pred,gt_label)
                    show1 = accuracy.data.cpu().numpy()
                    show2 = cls_loss.data.cpu().numpy()
                    show3 = box_offset_loss.data.cpu().numpy()
                    show4 = landmark_loss.data.cpu().numpy()
                    show5 = all_loss.data.cpu().numpy()
                    print("%s : Epoch: %d, Step: %d, accuracy: %s, det loss: %s, bbox loss: %s, landmark_loss: %s all_loss: %s, lr:%s "%
                            (datetime.datetime.now(),epoch,batch_idx, show1,show2,show3,show4,show5,run.lr))
                    epoch_acc = show1
                #计算偏差矩阵
                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()
                pass
            
            pass
            print('save modle acc:', epoch_acc)
            torch.save(network.state_dict(), os.path.join(model_store_path,"onet_epoch_%d.pt" % epoch))
            torch.save(network, os.path.join(model_store_path,"onet_epoch_model_%d.pkl" % epoch))
            pass
        
        pass
    
    pass              


# In[ ]:





# In[ ]:


if __name__ == '__main__':
    print('train Onet Process:...')
    #加载图片文件
    #imagedb = ImageDB(annotation_file,'./image/train')
    #gt_imdb = imagedb.load_imdb()
    #gt_imdb = imagedb.append_flipped_images(gt_imdb)
    train_net()
    print('finish....')
    #print(gt_imdb[2])

