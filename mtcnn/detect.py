#!/usr/bin/env python
# coding: utf-8

# ##### 导入头文件

# In[2]:


import sys
sys.path.append('../')
from mtcnn import PNet


# ##### 生成 mtcnn

# In[1]:


def create_mtcnn_pnet(p_model_path=None,use_cuda=False):
    pnet = None
    if p_model_path is not None:
        pnet = PNet()
        if(use_cuda):
            print('p_model_path:{0}'.format(p_model_path))
            pnet.load_state_dict(torch.load(p_model_path))
            pnet.cuda()
        else:
            # forcing all GPU tensors to be in CPU while loading
            pnet.load_state_dict(torch.load(p_model_path, map_location=lambda storage, loc: storage))
        pnet.eval() 
    return pnet


# ##### resize 图片大小

# In[ ]:


def resize_image(img, scale):
    """
        resize image and transform dimention to [batchsize, channel, height, width]
    Parameters:
    ----------
        img: numpy array , height x width x channel
            input image, channels in BGR order here
        scale: float number
            scale factor of resize operation
    Returns:
    -------
        transformed image tensor , 1 x channel x height x width
    """
    height, width, channels = img.shape
    new_height = int(height * scale)     # resized new height
    new_width = int(width * scale)       # resized new width
    new_dim = (new_width, new_height)
    img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)      # resized image
    return img_resized


# ##### 生成 bounding box

# In[ ]:


def generate_bounding_box(mp, reg, scale, threshold):
    """
        generate bbox from feature mp
    Parameters:
    ----------
        mp: numpy array , n x m x 1
            detect score for each position
        reg: numpy array , n x m x 4
            bbox
        scale: float number
            scale of this detection
        threshold: float number
            detect threshold
    Returns:
    -------
        bbox array
    """
    stride = 2

    cellsize = 12 # receptive field

    t_index = np.where(mp > threshold)
    # print('shape of t_index:{0}'.format(len(t_index)))
    # print('t_index{0}'.format(t_index))
    # time.sleep(5)
    # find nothing
    if t_index[0].size == 0:
        return np.array([])
    # reg = (1, n, m, 4)
    # choose bounding box whose socre are larger than threshold
    dx1, dy1, dx2, dy2 = [reg[0, t_index[0], t_index[1], i] for i in range(4)]
    # print(dx1.shape)
    # time.sleep(5)
    reg = np.array([dx1, dy1, dx2, dy2])
    # print('shape of reg{0}'.format(reg.shape))
    # lefteye_dx, lefteye_dy, righteye_dx, righteye_dy, nose_dx, nose_dy, \
    # leftmouth_dx, leftmouth_dy, rightmouth_dx, rightmouth_dy = [landmarks[0, t_index[0], t_index[1], i] for i in range(10)]
    #
    # landmarks = np.array([lefteye_dx, lefteye_dy, righteye_dx, righteye_dy, nose_dx, nose_dy, leftmouth_dx, leftmouth_dy, rightmouth_dx, rightmouth_dy])
    # abtain score of classification which larger than threshold
    # t_index[0]: choose the first column of t_index
    # t_index[1]: choose the second column of t_index
    score = mp[t_index[0], t_index[1], 0]
    # hence t_index[1] means column, t_index[1] is the value of x
    # hence t_index[0] means row, t_index[0] is the value of y
    boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),            # x1 of prediction box in original image
                             np.round((stride * t_index[0]) / scale),            # y1 of prediction box in original image
                             np.round((stride * t_index[1] + cellsize) / scale), # x2 of prediction box in original image
                             np.round((stride * t_index[0] + cellsize) / scale), # y2 of prediction box in original image                                                                         
                             score,                                              # reconstruct the box in original image
                             reg,
                             # landmarks
                             ])
    return boundingbox.T


# ##### 统一图片格式

# In[ ]:


def unique_image_format(im):
    if not isinstance(im,np.ndarray):
        if im.mode == 'I':
            im = np.array(im, np.int32, copy=False)
        elif im.mode == 'I;16':
            im = np.array(im, np.int16, copy=False)
        else:
            im = np.asarray(im)
    return im


# ##### PNet 检测器

# In[3]:


class PNetDetector(object):
    def __init__(self,pnet=None,min_face_size=12,stride=2,threshold=[0.6, 0.7, 0.7],scale_factor=0.709):
        self.pnet_detector = pnet
        self.min_face_size = min_face_size
        self.stride=stride
        self.thresh = threshold
        self.scale_factor = scale_factor
  
    def detect_pnet(self, im):
        """Get face candidates through pnet
        Parameters:
        ----------
        im: numpy array
            input image array
            one batch
        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_align: numpy array
            boxes after calibration
        """
        # im = self.unique_image_format(im)
        # original wider face data
        h, w, c = im.shape

        net_size = 12

        current_scale = float(net_size) / self.min_face_size    # find initial scale
        # print('imgshape:{0}, current_scale:{1}'.format(im.shape, current_scale))
        im_resized = resize_image(im, current_scale) 
        # scale = 1.0
        current_height, current_width, _ = im_resized.shape
        # fcn
        all_boxes = list()
        #i = 0
        while min(current_height, current_width) > net_size:
            # print(i)
            feed_imgs = []
            image_tensor = image_tools.convert_image_to_tensor(im_resized)
            feed_imgs.append(image_tensor)
            feed_imgs = torch.stack(feed_imgs)
            feed_imgs = Variable(feed_imgs)

            if self.pnet_detector.use_cuda:
                feed_imgs = feed_imgs.cuda()

            # self.pnet_detector is a trained pnet torch model
            # receptive field is 12×12
            # 12×12 --> score
            # 12×12 --> bounding box
            cls_map, reg = self.pnet_detector(feed_imgs)
            
            cls_map_np = image_tools.convert_chwTensor_to_hwcNumpy(cls_map.cpu())
            
            reg_np = image_tools.convert_chwTensor_to_hwcNumpy(reg.cpu())
            # print(cls_map_np.shape, reg_np.shape) # cls_map_np = (1, n, m, 1) reg_np.shape = (1, n, m 4)
            # time.sleep(5)
            # landmark_np = image_tools.convert_chwTensor_to_hwcNumpy(landmark.cpu())
            # self.threshold[0] = 0.6
            # print(cls_map_np[0,:,:].shape)
            # time.sleep(4)
            # boxes = [x1, y1, x2, y2, score, reg]
            boxes = self.generate_bounding_box(cls_map_np[ 0, :, :], reg_np, current_scale, self.thresh[0])
            # generate pyramid images
            current_scale *= self.scale_factor 
            # self.scale_factor = 0.709
            im_resized = self.resize_image(im, current_scale)
            
            current_height, current_width, _ = im_resized.shape

            if boxes.size == 0:
                continue

            # non-maximum suppresion
            keep = utils.nms(boxes[:, :5], 0.5, 'Union')
            boxes = boxes[keep]
            # print(boxes.shape)
            all_boxes.append(boxes)
            # i+=1

        if len(all_boxes) == 0:
            return None, None

        all_boxes = np.vstack(all_boxes)
        # print("shape of all boxes {0}".format(all_boxes.shape))
        # time.sleep(5)

        # merge the detection from first stage
        keep = utils.nms(all_boxes[:, 0:5], 0.7, 'Union')
        all_boxes = all_boxes[keep]
        # boxes = all_boxes[:, :5]

        # x2 - x1
        # y2 - y1
        #bw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        #bh = all_boxes[:, 3] - all_boxes[:, 1] + 1
        
        bw = all_boxes[:, 2] 
        
        bh = all_boxes[:, 3]

        # landmark_keep = all_boxes[:, 9:].reshape((5,2))

        boxes = np.vstack([all_boxes[:,0],
                   all_boxes[:,1],
                   all_boxes[:,2],
                   all_boxes[:,3],
                   all_boxes[:,4],
                   # all_boxes[:, 0] + all_boxes[:, 9] * bw,
                   # all_boxes[:, 1] + all_boxes[:,10] * bh,
                   # all_boxes[:, 0] + all_boxes[:, 11] * bw,
                   # all_boxes[:, 1] + all_boxes[:, 12] * bh,
                   # all_boxes[:, 0] + all_boxes[:, 13] * bw,
                   # all_boxes[:, 1] + all_boxes[:, 14] * bh,
                   # all_boxes[:, 0] + all_boxes[:, 15] * bw,
                   # all_boxes[:, 1] + all_boxes[:, 16] * bh,
                   # all_boxes[:, 0] + all_boxes[:, 17] * bw,
                   # all_boxes[:, 1] + all_boxes[:, 18] * bh
                  ])

        boxes = boxes.T

        # boxes = boxes = [x1, y1, x2, y2, score, reg] reg= [px1, py1, px2, py2] (in prediction)
        # boxes = boxes = [x1, y1, w, h, score, reg] reg= [px1, py1, px2, py2] (in prediction)
        align_topx = all_boxes[:, 0] + all_boxes[:, 5] * bw
        align_topy = all_boxes[:, 1] + all_boxes[:, 6] * bh
        
        align_bottomx = all_boxes[:, 0] + all_boxes[:, 2] - 1 + all_boxes[:, 7] * bw
        align_bottomy = all_boxes[:, 1] + all_boxes[:, 3] - 1 + all_boxes[:, 8] * bh

        # refine the boxes
        boxes_align = np.vstack([ align_topx,
                              align_topy,
                              align_bottomx,
                              align_bottomy,
                              all_boxes[:, 4],
                              # align_topx + all_boxes[:,9] * bw,
                              # align_topy + all_boxes[:,10] * bh,
                              # align_topx + all_boxes[:,11] * bw,
                              # align_topy + all_boxes[:,12] * bh,
                              # align_topx + all_boxes[:,13] * bw,
                              # align_topy + all_boxes[:,14] * bh,
                              # align_topx + all_boxes[:,15] * bw,
                              # align_topy + all_boxes[:,16] * bh,
                              # align_topx + all_boxes[:,17] * bw,
                              # align_topy + all_boxes[:,18] * bh,
                              ])
        
        boxes_align = boxes_align.T

        return boxes, boxes_align


# In[ ]:




