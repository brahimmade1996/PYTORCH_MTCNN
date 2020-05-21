#!/usr/bin/env python
# coding: utf-8

# #### Process data

# In[1]:


import os
from scipy.io import loadmat

class DATA:
    def __init__(self, image_name,bboxes):
        self.image_name = image_name
        #self.facenumber = 
        self.bboxes = bboxes


class WIDER(object):
    def __init__(self, file_to_label, path_to_image=None):
        self.file_to_label = file_to_label
        self.path_to_image = path_to_image

        self.f = loadmat(file_to_label)
        self.event_list = self.f['event_list']
        self.file_list = self.f['file_list']
        self.face_bbx_list = self.f['face_bbx_list']

    def next(self):
        for event_idx, event in enumerate(self.event_list):
            # fix error of "can't not .. bytes and strings"
            e = str(event[0][0].encode('utf-8'))[2:-1]
            for file, bbx in zip(self.file_list[event_idx][0],
                                 self.face_bbx_list[event_idx][0]):
                f = file[0][0].encode('utf-8')
                #print(e, f)  # bytes, bytes
                # fix error of "can't not .. bytes and strings"
                f = str(f)[2:-1]
                # path_of_image = os.path.join(self.path_to_image, str(e), str(f)) + ".jpg"
                path_of_image = self.path_to_image + '/' + e + '/' + f + ".jpg"
                # print(path_of_image)
                bboxes = []
                bbx0 = bbx[0]
                for i in range(bbx0.shape[0]):
                    xmin, ymin, xmax, ymax = bbx0[i]
                    bboxes.append((int(xmin), int(ymin), int(xmax), int(ymax)))
                yield DATA(path_of_image, bboxes)
                    


# ##### WIDER_FACE MAT FORMAT

# In[5]:


import scipy.io as scio

path = '../image/wider_annotation/wider_face_train.mat'
reftracker = scio.loadmat(path)

print(list(reftracker.keys()))


# ##### MAKE DATA

# In[6]:


import os
import sys
sys.path.append(os.getcwd())
import cv2
import time

"""
 modify .mat to .txt 
"""

#wider face original images path
path_to_image = '../image/wider_train/images'

#matlab file path
file_to_label = '../image/wider_annotation/wider_face_train.mat'

#target file path
target_file = '../image/anno_train.txt'

wider = WIDER(file_to_label,path_to_image)

line_count = 0
box_count = 0

print('start transforming....')
t = time.time()

with open(target_file, 'w+') as f:
    # press ctrl-C to stop the process
    for data in wider.next():
        line = []
        line.append(str(data.image_name))
        line_count += 1
        for i,box in enumerate(data.bboxes):
            box_count += 1
            for j,bvalue in enumerate(box):
                line.append(str(bvalue))

        line.append('\n')

        line_str = ' '.join(line)
        f.write(line_str)

st = time.time()-t
print('end transforming')

print('spend time:%d'%st)
print('total line(images):%d'%line_count)
print('total boxes(faces):%d'%box_count)


# In[ ]:




