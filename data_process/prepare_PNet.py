#!/usr/bin/env python
# coding: utf-8

# #### Prepare PNet 

# In[2]:


def IoU(box, boxes):
    """Compute IoU between detect box and gt boxes
    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes
    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    # box = (x1, y1, x2, y2)
    # box = (x1, y1, w, h)
    #box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    box_area = box[2] * box[3]
    #area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    area = boxes[:,2]*boxes[:,3]
    # abtain the offset of the interception of union between crop_box and gt_box
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    #xx2 = np.minimum(box[2], boxes[:, 2])
    #yy2 = np.minimum(box[3], boxes[:, 3])
    
    xx2 = np.minimum(box[0]+box[2]-1, boxes[:, 2]+boxes[:,0]-1)
    yy2 = np.maximum(box[1]+box[3]-1, boxes[:, 1]+boxes[:,3]-1)
    
    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)

    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr


def convert_to_square(bbox):
    """Convert bbox to square
    Parameters:
    ----------
    bbox: numpy array , shape n x 5
        input bbox
    Returns:
    -------
    square bbox
    """
    square_bbox = bbox.copy()

    #h = bbox[:, 3] - bbox[:, 1] + 1
    #w = bbox[:, 2] - bbox[:, 0] + 1
    
    h = bbox[:,3]
    
    w = bbox[:,2]
    
    max_side = np.maximum(h,w)
    square_bbox[:, 0] = bbox[:, 0] + w*0.5 - max_side*0.5
    square_bbox[:, 1] = bbox[:, 1] + h*0.5 - max_side*0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
    square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
    return 


# #### code of create PNet data

# In[3]:


"""
    2018-10-20 15:50:20
    generate positive, negative, positive images whose size are 12*12 and feed into PNet
"""
import sys
import numpy as np
import cv2
import os
sys.path.append(os.getcwd())
import numpy as np

prefix = ''
anno_file = '../image/anno_train.txt'
im_dir = '../image/wider_train/images'
pos_save_dir = "../image/12/positive"
part_save_dir = "../image/12/part"
neg_save_dir = '../image/12/negative'

if not os.path.exists(pos_save_dir):
    os.mkdir(pos_save_dir)
if not os.path.exists(part_save_dir):
    os.mkdir(part_save_dir)
if not os.path.exists(neg_save_dir):
    os.mkdir(neg_save_dir)

# store labels of positive, negative, part images
f1 = open(os.path.join('../image', 'pos_12.txt'), 'w')
f2 = open(os.path.join('../image', 'neg_12.txt'), 'w')
f3 = open(os.path.join('../image', 'part_12.txt'), 'w')

# anno_file: store labels of the wider face training data
with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)
print("%d pics in total" % num)

p_idx = 0 # positive
n_idx = 0 # negative
d_idx = 0 # dont care
idx = 0
box_idx = 0

for annotation in annotations:
    annotation = annotation.strip().split(' ')
    im_path = os.path.join(prefix, annotation[0])
    #print(im_path)
    bbox = list(map(float, annotation[1:]))
    boxes = np.array(bbox, dtype=np.int32).reshape(-1, 4)
    img = cv2.imread(im_path)
    idx += 1
    if idx % 1000 == 0:
        break
    height, width, channel = img.shape

    neg_num = 0
    while neg_num < 50:
        size = np.random.randint(12, min(width, height) / 2)
        nx = np.random.randint(0, width - size)
        ny = np.random.randint(0, height - size)
        crop_box = np.array([nx, ny, nx + size, ny + size])

        Iou = IoU(crop_box, boxes)

        cropped_im = img[ny: ny + size, nx: nx + size, :]
        resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

        if np.max(Iou) < 0.3:
            # Iou with all gts must below 0.3
            save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
            f2.write(save_file + ' 0\n')
            cv2.imwrite(save_file, resized_im)
            n_idx += 1
            neg_num += 1

    for box in boxes:
        # box (x_left, y_top, x_right, y_bottom)
        x1, y1, w, h = box
        # w = x2 - x1 + 1
        # h = y2 - y1 + 1
        x2 = x1 + w - 1
        y2 = y1 + h - 1
        # ignore small faces
        # in case the ground truth boxes of small faces are not accurate
        if max(w, h) < 40 or x1 < 0 or y1 < 0:
            continue

        # generate negative examples that have overlap with gt
        #new_box = [x1,y1,x2,y2]
        for i in range(5):
            size = np.random.randint(12, min(width, height) / 2)
            # delta_x and delta_y are offsets of (x1, y1)

            delta_x = np.random.randint(max(-size, -x1), w)
            delta_y = np.random.randint(max(-size, -y1), h)
            nx1 = max(0, x1 + delta_x)
            ny1 = max(0, y1 + delta_y)

            if nx1 + size > width or ny1 + size > height:
                continue
            crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
            Iou = IoU(crop_box, boxes)
            #Iou = IoU(crop_box, new_box)
            
            cropped_im = img[ny1: ny1 + size, nx1: nx1 + size, :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                f2.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1

        # generate positive examples and part faces
        for i in range(20):
            size = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

            # delta here is the offset of box center
            delta_x = np.random.randint(-w * 0.2, w * 0.2)
            delta_y = np.random.randint(-h * 0.2, h * 0.2)

            nx1 = max(x1 + w / 2 + delta_x - size / 2, 0)
            ny1 = max(y1 + h / 2 + delta_y - size / 2, 0)
            nx2 = nx1 + size
            ny2 = ny1 + size

            if nx2 > width or ny2 > height:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])

            offset_x1 = (x1 - nx1) / float(size)
            offset_y1 = (y1 - ny1) / float(size)
            offset_x2 = (x2 - nx2) / float(size)
            offset_y2 = (y2 - ny2) / float(size)

            cropped_im = img[int(ny1): int(ny2), int(nx1): int(nx2), :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
            box_ = box.reshape(1, -1)
            #new_box = np.array([x1, y1, x2, y2])
            #box_ = new_box.reshape(1, -1)
            if IoU(crop_box, box_) >= 0.65:
                save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                f1.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                #print('postive:',save_file)
                p_idx += 1
            elif IoU(crop_box, box_) >= 0.4:
                save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                #print('postive:', save_file)
                d_idx += 1
        
        box_idx += 1
        pass
    print("%s images done, pos: %s part: %s neg: %s" % (idx, p_idx, d_idx, n_idx))
    pass

f1.close()
f2.close()
f3.close()

print('finish....')


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


# In[5]:


import os
import sys
sys.path.append(os.getcwd())

pnet_postive_file = '../image/pos_12.txt'
pnet_part_file = '../image/part_12.txt'
pnet_neg_file = '../image/neg_12.txt'
pnet_landmark_file = '../image/landmark_12.txt'
imglist_filename = '../image/imglist_anno_12.txt'

if __name__ == '__main__':

    anno_list = []

    anno_list.append(pnet_postive_file)
    anno_list.append(pnet_part_file)
    anno_list.append(pnet_neg_file)
    # anno_list.append(pnet_landmark_file)

    chose_count = assemble_data(imglist_filename ,anno_list)
    print("PNet train annotation result file path:%s" % imglist_filename)


# In[ ]:




