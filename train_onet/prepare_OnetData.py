#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[4]:


import argparse
import cv2
import numpy as np
import os
import sys
sys.path.append('../')
#from mtcnn.core.detect import MtcnnDetector,create_mtcnn_net
from tool.imagedb import ImageDB
from tool.imagedb import TestImageLoader
from tool.utils import IoU
import time
from six.moves import cPickle
from mtcnn import PNet
from mtcnn import RNet
from mtcnn import detect


# #### Generate train data of ONet

# In[5]:


def gen_onet_data(data_dir, anno_file, pnet_model_file, rnet_model_file, prefix_path='', use_cuda=True, vis=False):

    #pnet, rnet, _ = create_mtcnn_net(p_model_path=pnet_model_file, r_model_path=rnet_model_file, use_cuda=use_cuda)
    
    #mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, min_face_size=12)
    
    pnet = detect.create_mtcnn_pnet(p_model_path=pnet_model_file)
    
    pnet_detect = detect.PNetDetector(pnet)
    
    rnet = detect.create_mtcnn_rnet(r_model_path=rnet_model_file)
    
    rnet_detect = detect.RNetDetector(rnet)
    

    imagedb = ImageDB(anno_file,mode="test",prefix_path=prefix_path)
    imdb = imagedb.load_imdb()
    image_reader = TestImageLoader(imdb,1,False)

    all_boxes = list()
    
    batch_idx = 0
    print('size:%d' % image_reader.size)
    
    for databatch in image_reader:
        if batch_idx % 50 == 0:
            print("%d images done" % batch_idx)
            
        im = databatch
        t = time.time()
        # pnet detection = [x1, y1, x2, y2, score, reg]
        p_boxes, p_boxes_align = mtcnn_detector.detect_pnet(im=im)
        # rnet detection
        boxes, boxes_align = mtcnn_detector.detect_rnet(im=im, dets=p_boxes_align)

        if boxes_align is None:
            all_boxes.append(np.array([]))
            batch_idx += 1
            continue
        if vis:
            rgb_im = cv2.cvtColor(np.asarray(im), cv2.COLOR_BGR2RGB)
            vision.vis_two(rgb_im, boxes, boxes_align)

        t1 = time.time() - t
        t = time.time()
        all_boxes.append(boxes_align)
        batch_idx += 1

    save_path = '../image/48'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_file = os.path.join(save_path, "detections_%d.pkl" % int(time.time()))
    with open(save_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)
    gen_onet_sample_data(data_dir,anno_file,save_file,prefix_path)


# #### Define a Path of model of PNet and RNet

# In[6]:


prefix_path = ''
traindata_store = ''
pnet_model_file = '../model/Pnet/pnet_epoch_9.pt'
rnet_model_file = '../model/Rnet/rnet_epoch_9.pt'
annotation_file = '../image/anno_train.txt'
use_cuda = True
gen_onet_data(traindata_store, annotation_file, pnet_model_file, rnet_model_file, prefix_path, use_cuda)


# #### Define function of collect image data

# In[ ]:


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


# #### collect train data

# In[ ]:


import os
import sys
sys.path.append(os.getcwd())

onet_postive_file = '../image/48/pos_48.txt'
onet_part_file = '../image/48/part_48.txt'
onet_neg_file = '../image/48/neg_48.txt'
onet_landmark_file = '../image/48/landmark_48.txt'
imglist_filename = '../image/48/imglist_anno_48.txt'

if __name__ == '__main__':

    anno_list = []

    anno_list.append(onet_postive_file)
    anno_list.append(onet_part_file)
    anno_list.append(onet_neg_file)
    anno_list.append(onet_landmark_file)
    
    chose_count = assemble_data(imglist_filename ,anno_list)
    print("ONet train annotation result file path:%s" % imglist_filename)

