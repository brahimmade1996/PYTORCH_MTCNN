#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np

class ImageDB(object):
    def __init__(self, image_annotation_file, prefix_path='', mode='train'):
        self.prefix_path = prefix_path
        self.image_annotation_file = image_annotation_file
        self.classes = ['__background__', 'face']
        self.num_classes = 2
        self.image_set_index = self.load_image_set_index()
        self.num_images = len(self.image_set_index)
        self.mode = mode


    def load_image_set_index(self):
        """Get image index
        Parameters:
        ----------
        Returns:
        -------
        image_set_index: str
            relative path of image
        """
        assert os.path.exists(self.image_annotation_file), 'Path does not exist: {}'.format(self.image_annotation_file)
        with open(self.image_annotation_file, 'r') as f:
            image_set_index = [x.strip().split(' ')[0] for x in f.readlines()]
        return image_set_index


    def load_imdb(self):
        """Get and save ground truth image database
        Parameters:
        ----------
        Returns:
        -------
        gt_imdb: dict
            image database with annotations
        """
        #cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        #if os.path.exists(cache_file):
        #    with open(cache_file, 'rb') as f:
        #        imdb = cPickle.load(f)
        #    print '{} gt imdb loaded from {}'.format(self.name, cache_file)
        #    return imdb

        gt_imdb = self.load_annotations()

        #with open(cache_file, 'wb') as f:
        #    cPickle.dump(gt_imdb, f, cPickle.HIGHEST_PROTOCOL)
        return gt_imdb


    def real_image_path(self, index):
        """Given image index, return full path
        Parameters:
        ----------
        index: str
            relative path of image
        Returns:
        -------
        image_file: str
            full path of image
        """

        index = index.replace("\\", "/")

        if not os.path.exists(index):
            image_file = os.path.join(self.prefix_path, index)
        else:
            image_file=index
        if not image_file.endswith('.jpg'):
            image_file = image_file + '.jpg'
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file


    def load_annotations(self,annotion_type=1):
        """Load annotations
        Parameters:
        ----------
        annotion_type: int
                      0:dsadsa
                      1:dsadsa
        Returns:
        -------
        imdb: dict
            image database with annotations
        """

        assert os.path.exists(self.image_annotation_file), 'annotations not found at {}'.format(self.image_annotation_file)
        with open(self.image_annotation_file, 'r') as f:
            annotations = f.readlines()

        imdb = []
        for i in range(self.num_images):
            annotation = annotations[i].strip().split(' ')
            #print(len(annotation))
            index = annotation[0]
            im_path = self.real_image_path(index)
            imdb_ = dict()
            imdb_['image'] = im_path

            if self.mode == 'test':
               # gt_boxes = map(float, annotation[1:])
               # boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
               # imdb_['gt_boxes'] = boxes
                pass
            else:
                label = annotation[1]
                imdb_['label'] = int(label)
                imdb_['flipped'] = False
                imdb_['bbox_target'] = np.zeros((4,))
                imdb_['landmark_target'] = np.zeros((10,))
                
                if len(annotation[2:])==4:
                    #print(len(annotation[2:]))
                    bbox_target = annotation[2:6]
                    imdb_['bbox_target'] = np.array(bbox_target).astype(float)
                if len(annotation[2:])==14:
                    bbox_target = annotation[2:6]
                    imdb_['bbox_target'] = np.array(bbox_target).astype(float)
                    landmark = annotation[6:]
                    imdb_['landmark_target'] = np.array(landmark).astype(float)
            imdb.append(imdb_)

        return imdb


    def append_flipped_images(self, imdb):
        """append flipped images to imdb
        Parameters:
        ----------
        imdb: imdb
            image database
        Returns:
        -------
        imdb: dict
            image database with flipped image annotations added
        """
        print('append flipped images to imdb', len(imdb))
        for i in range(len(imdb)):
            imdb_ = imdb[i]
            m_bbox = imdb_['bbox_target'].copy()
            m_bbox[0], m_bbox[2] = -m_bbox[2], -m_bbox[0]

            landmark_ = imdb_['landmark_target'].copy()
            landmark_ = landmark_.reshape((5, 2))
            landmark_ = np.asarray([(1 - x, y) for (x, y) in landmark_])
            landmark_[[0, 1]] = landmark_[[1, 0]]
            landmark_[[3, 4]] = landmark_[[4, 3]]

            item = {'image': imdb_['image'],
                     'label': imdb_['label'],
                     'bbox_target': m_bbox,
                     'landmark_target': landmark_.reshape((10)),
                     'flipped': True}

            imdb.append(item)
        self.image_set_index *= 2
        return imdb


# In[2]:


#imagedb = ImageDB('./image/train/trainImageList.txt','./image/train')
#gt_imdb = imagedb.load_imdb()
#gt_imdb = imagedb.append_flipped_images(gt_imdb)
#print(gt_imdb[0])


# In[3]:


import numpy as np
import cv2



class TrainImageReader:
    def __init__(self, imdb, im_size, batch_size=128, shuffle=False):

        self.imdb = imdb
        self.batch_size = batch_size
        self.im_size = im_size
        self.shuffle = shuffle

        self.cur = 0
        self.size = len(imdb)
        self.index = np.arange(self.size)
        self.num_classes = 2

        self.batch = None
        self.data = None
        self.label = None

        self.label_names= ['label', 'bbox_target', 'landmark_target']
        self.reset()
        self.get_batch()

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return self.data,self.label
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        imdb = [self.imdb[self.index[i]] for i in range(cur_from, cur_to)]
        data, label = get_minibatch(imdb)
        self.data = data['data']
        self.label = [label[name] for name in self.label_names]



class TestImageLoader:
    def __init__(self, imdb, batch_size=1, shuffle=False):
        self.imdb = imdb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.size = len(imdb)
        self.index = np.arange(self.size)

        self.cur = 0
        self.data = None
        self.label = None

        self.reset()
        self.get_batch()

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return self.data
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        imdb = [self.imdb[self.index[i]] for i in range(cur_from, cur_to)]
        data= get_testbatch(imdb)
        self.data=data['data']




def get_minibatch(imdb):

    # im_size: 12, 24 or 48
    num_images = len(imdb)
    processed_ims = list()
    cls_label = list()
    bbox_reg_target = list()
    landmark_reg_target = list()

    for i in range(num_images):
        im = cv2.imread(imdb[i]['image'])
        #im = Image.open(imdb[i]['image'])

        if imdb[i]['flipped']:
            im = im[:, ::-1, :]
            #im = im.transpose(Image.FLIP_LEFT_RIGHT)

        cls = imdb[i]['label']
        bbox_target = imdb[i]['bbox_target']
        landmark = imdb[i]['landmark_target']

        processed_ims.append(im)
        cls_label.append(cls)
        bbox_reg_target.append(bbox_target)
        landmark_reg_target.append(landmark)

    im_array = np.asarray(processed_ims)

    label_array = np.array(cls_label)

    bbox_target_array = np.vstack(bbox_reg_target)

    landmark_target_array = np.vstack(landmark_reg_target)

    data = {'data': im_array}
    label = {'label': label_array,
             'bbox_target': bbox_target_array,
             'landmark_target': landmark_target_array
             }

    return data, label


def get_testbatch(imdb):
    assert len(imdb) == 1, "Single batch only"
    im = cv2.imread(imdb[0]['image'])
    data = {'data': im}
    return data


# In[ ]:




