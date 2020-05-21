#!/usr/bin/env python
# coding: utf-8

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

import pdb

from collections import OrderedDict
from collections import namedtuple
from itertools import product

#torch.set_printoptions(linewidth=120)
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias, 0.1)


# #### 定义神经网络
# 
# 1. 定义模型属性
# 2. 定义forward函数

# #### 定义PNet
# 
# 1. 输入图片 12*12*3
# 2. 三个输出代表：人脸分类、人脸框的回归和人脸关键点定位
# 3. bound的4个坐标信息和score 4*1*1

# In[2]:




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
        label = F.sigmoid(det)
        #out bounding box (1*1*4)
        bound = self.bound(t)
        offset = F.relu(bound)
        #landmark = self.landmark(t)
        return label,offset
    pass

"""



"""
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

from sklearn.metrics import confusion_matrix


# #### 定义RNet
# 
# 1. R-Net主要用来去除大量的非人脸框
# 
# 2. 输入是PNet的bound矩阵 24*24*3大小 可以resize Pnet 的offset
# 

# In[3]:


#input 24*24*3
class RNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=28,kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=28,out_channels=48,kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=48,out_channels=64,kernel_size=2)
        self.out = nn.Linear(in_features=3*3*64,out_features=128)
        
        self.det = nn.Linear(in_features=128,out_features=2)
        self.bound = nn.Linear(in_features=128,out_features=4)
        self.landmark = nn.Linear(in_features=128,out_features=10)
        pass
    
    def forward(self,tensor):
        #layer input
        input = tensor
        #layer 1
        t=self.conv1(input)
        t=F.relu(t)
        #print('RNet conv1 shape:',t.shape)
        t=F.max_pool2d(t,kernel_size=3,stride=2)
        #print('RNet MP1 3*3 shape:',t.shape)
        #layer 2
        t=self.conv2(t)
        t=F.relu(t)
        #print('RNet conv2 shape: ',t.shape)
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
        print('Rnet out label shape:',label.shape)
        #out bounding box (B*4)
        bound = self.bound(t)
        offset = F.relu(bound)
        print('Rnet out offset shape:',offset.shape)
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


# #### 定义ONet
# 
# 1. ONet和RNet有些像，只不过这一步还增加了landmark位置的回归
# 
# 2. 输入大小调整为48*48 ,RNet输出的Bound即offset
# 
# 
# 

# In[4]:


#input 48*48*3
class ONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=2)
        self.out = nn.Linear(in_features=3*3*128,out_features=256)
        
        self.det = nn.Linear(in_features=256,out_features=2)
        self.bound = nn.Linear(in_features=256,out_features=4)
        self.landmark = nn.Linear(in_features=256,out_features=10)
        pass
    def forward(self,tensor):
        #layer input
        input = tensor
        #layer 1
        t=self.conv1(input)
        t=F.relu(t)
        #print('ONet conv1 shape:',t.shape)
        t=F.max_pool2d(t,kernel_size=2,stride=2)
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
        t=t.reshape(-1,128*3*3)
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


# #### 训练模型 建立标准函数

# In[5]:


def ger_num_correct(preds,labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

@torch.no_grad()
def ger_all_preds(model,train_data):
    all_preds=torch.tensor([])
    for batch in train_data:
        images,labels = batch
        preds=model(images)
        torch.cat((all_preds,preds),dim=0)
    return all_preds


# #### 计算混淆矩阵

# In[14]:


class RunBuilder():
    @staticmethod
    def get_runs(params):

        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs


# #### 保存矩阵

# In[18]:


class Epoch():
    def __init__(self):
        self.count = 0
        self.loss = 0
        self.num_correct = 0
        self.start_time = None

class RunManager():
    def __init__(self):

        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None

        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None

        self.network = None
        self.loader = None
        self.tb = None
        pass

    def begin_run(self, run, network, loader):

        self.run_start_time = time.time()

        self.run_params = run
        self.run_count += 1

        self.network = network
        self.loader = loader
        self.tb = SummaryWriter(comment=f'-{run}')

        images, labels = next(iter(self.loader))
        grid = torchvision.utils.make_grid(images)

        self.tb.add_image('images', grid)
        self.tb.add_graph(self.network, images)
        
    def end_run(self):
        self.tb.close()
        self.epoch_count = 0
    
    def begin_epoch(self):
        self.epoch_start_time = time.time()

        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        
    def end_epoch(self):

        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss = self.epoch_loss / len(self.loader.dataset)
        accuracy = self.epoch_num_correct / len(self.loader.dataset)

        self.tb.add_scalar('Loss', loss, self.epoch_count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)

        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)
            
        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results['loss'] = loss
        results["accuracy"] = accuracy
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        for k,v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)
        df = pd.DataFrame.from_dict(self.run_data, orient='columns')
        
        clear_output(wait=True)
        display(df)
        
    def track_loss(self, loss):
        self.epoch_loss += loss.item() * self.loader.batch_size

    def track_num_correct(self, preds, labels):
        self.epoch_num_correct += self.get_num_correct(preds, labels)
        
    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()
    
    def save(self, fileName):
        pd.DataFrame.from_dict(
            self.run_data, orient='columns'
        ).to_csv(f'{fileName}.csv')
        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)
    pass


# #### 计算损失函数

# In[1]:


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


# In[4]:


import os
import numpy.random as npr
import numpy as np

def assemble_data(output_file, anno_file_list=[]):

    #assemble the pos, neg, part annotations to one file
    size = 12

    if len(anno_file_list)==0:
        return 0

    if os.path.exists(output_file):
        os.remove(output_file)

    for anno_file in anno_file_list:
        with open(anno_file, 'r') as f:
            print(anno_file)
            anno_lines = f.readlines()

        base_num = 250000

        if len(anno_lines) > base_num * 3:
            idx_keep = npr.choice(len(anno_lines), size=base_num * 3, replace=True)
        elif len(anno_lines) > 100000:
            idx_keep = npr.choice(len(anno_lines), size=len(anno_lines), replace=True)
        else:
            idx_keep = np.arange(len(anno_lines))
            np.random.shuffle(idx_keep)
        chose_count = 0
        with open(output_file, 'a+') as f:
            for idx in idx_keep:
                # write lables of pos, neg, part images
                f.write(anno_lines[idx])
                chose_count+=1

    return chose_count

