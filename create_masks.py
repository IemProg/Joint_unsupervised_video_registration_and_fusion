#semantic segmentation

# example of inference with a pre-trained coco model
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2gray
from tqdm import tqdm
from utils import *

#Local Imports
sys.path.append('./Mask_RCNN/')

#3rd Party Libraries
from mrcnn.config import Config
from mrcnn.model import MaskRCNN

# define the test configuration
class TestConfig(Config):
     NAME = "test"
     GPU_COUNT = 1
     IMAGES_PER_GPU = 1
     NUM_CLASSES = 1 + 80  #Background + Person


# define 81 classes that the coco model knowns about
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

#Define the model
rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())

#Load coco model weights
print("################# Loading Weights #################")
rcnn.load_weights('./Mask_RCNN/mask_rcnn_coco.h5', by_name=True)

print("################# Loading Datasets #################")
#Working on visible data  SAVE_1_ir0_AFI.zip
path_ir1 = "/ivrldata1/students/imad/data/SAVE_1_ir0_AFI"
path_vis1 = "/ivrldata1/students/imad/data/SAVE_1_visible_AFI"

#Working on visible data  SAVE_1_ir0_AFI.zip
path_ir2 = "/ivrldata1/students/imad/data/SAVE_2_ir0_AFI"
path_vis2 = "/ivrldata1/students/imad/data/SAVE_2_visible_AFI"

#Working on visible data  SAVE_2_ir0_HAK.zip
path_ir3 = "/ivrldata1/students/imad/data/SAVE_2_ir0_HAK"
path_vis3 = "/ivrldata1/students/imad/data/SAVE_2_visible_HAK"

#Working on visible data  SAVE_4_ir0_ARO.zip
path_ir4 = "/ivrldata1/students/imad/data/SAVE_4_ir0_ARO"
path_vis4 = "/ivrldata1/students/imad/data/SAVE_4_visible_ARO"

#To get rid of the comments of TF
os.system("clear")

#visible, ir = dataload_rgb(path_vis2, path_ir2)

visible = dataload_rgb(path_vis2)
#ir = dataload_seg(path_ir2)

#visible = visible.astype('uint8')
#ir = ir.astype('uint8')

print("Visible: {}: ".format(visible.shape))
#print("IR: {}: ".format(ir.shape))


#Saving
nom_video_ir = "SAVE_2_ir0_AFI"
nom_video_vis= "SAVE_2_visible_AFI"
save_path_vis = "/ivrldata1/students/imad/data/Segmented/" + nom_video_vis
save_path_ir = "/ivrldata1/students/imad/data/Segmented/" + nom_video_ir

def segmentation_infrared(imgs):
    #Output array : is sample * H * W * 1
    output_array = np.zeros((imgs.shape[0], imgs.shape[1], imgs.shape[2]))
    output_array = output_array.astype(float)

    #for i in tqdm(range(imgs.shape[0])):
    for i in tqdm(range(imgs.shape[0])):
        binary, seg_img = segmentation_ir(imgs[i])
        output_array[i] = binary
    return output_array

def segmentation_rcnn(imgs):
     #for i in tqdm(range(imgs.shape[0])):
     for i in tqdm(range(imgs.shape[0])):
          seg_img = rcnn.detect([imgs[i]], verbose=0)[0]
          labels = [class_names[ele] for ele in seg_img['class_ids']]

          sample = np.zeros((imgs.shape[1], imgs.shape[2]))
          sample = sample.astype(float)
          #print("Mask shape for {} is: {}".format(i, seg_img['masks'].shape))
          #sample = seg_img['masks'][:, :, 0]
          if seg_img['masks'].shape[2] != 0:
               for k in range(seg_img['masks'].shape[2]):
                    #print("Class ID: {}".format(labels[k]))
                    if labels[k] == 'person':
                         sample += seg_img['masks'][:, :, k]
                         #print("Mask RGB {} added as person!".format(k))
          else:
               sample = np.zeros((imgs.shape[1], imgs.shape[2]))

          sample = sample.astype("float")
          sample = sample * 255
          cv2.imwrite(save_path_vis + "/frame" + str(i)+".png", sample)

     print("Done Creating Masks")

segmentation_rcnn(visible)
#segmentation_infrared(imgs)