#!/usr/bin/python

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import AffineNet
from torch.distributions.normal import Normal

from skimage.transform import resize
from skimage import io
from skimage import color
from imageio import imread, imwrite

import argparse
from utils import *
from model import AffineNet

sys.path.append('./fusion/')

from filters import *
from torchvision.models.vgg import vgg19
import vggfusiongpu as fu

#vgg_model = vgg19(True).cuda().eval()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--inputVid1", required=True,
    help="path to input RGB Frames")

ap.add_argument("-i", "--inputVid2", required=True,
    help="path to input IR Frames")

ap.add_argument("-i", "--inputSegFrame1", required=True,
    help="path to input RGB Frames")

ap.add_argument("-i", "--inputSegFrame2", required=True,
    help="path to input IR Frames")

ap.add_argument("-o", "--output", required=True,
    help="Output path to save output registred frames")

ap.add_argument("-m", "--model", type=str, required=True,
    help="Pretrained Model name for registration")

ap.add_argument("-s", "--save", type=str,  default = False, required=False,
    help="If you would like to save samples")

args = vars(ap.parse_args())

#print(args)

vis_path = args['inputVid1']
ir_path = args['inputVid2']
vis_seg_path = args['inputSegFrame1']
ir_seg_path = args['inputSegFrame2']

saving_path = args['output']
model_name = args['model']
save_samples = args['save']


#TO-DO List:
# [X] 1 - Load input videos into 2-arrays
# [X] 2 - Segmentation for Images                               DONE
# [X] 3 - Registration of Masks based on Pre-trained model
# [X] 4 - Interpolation of Masks on Origanal images
# [ ] 5 - Fusion of IR + RGB
# [ ] 6 - Color Blending

#Requirements : - VGG-19, Mask-RCNN

# Input : IR Videos, RGB, Registration Model of Masks, Fusion
# 		  It requires too much time, therefore video should be short
# output: Video: .mp4


"""
Inputs: 1st argument : IR_Array (N*D*3) or it Could be a folde of Images
		2nd argument : RGB_Array (N*D*3), 
		3rd argument : Pre-trained Model of Registration,  
		4th argument : IMG_PER_GPU, 
		5th argument : GPU_COUNT
		6th argument : Save Segmentation
"""
#To get rid of the comments of TF
os.system("clear")

#################################################
##			       Data Loading 		       ##
#################################################
vis = dataload(vis_path)
ir = dataload(ir_path)

visible_seg = dataload(vis_seg_path)
ir_seg = dataload(ir_seg_path)

print("Max of IR segmented: ", ir_seg.max())
print("Max of VISIBLE segmented: ", visible_seg.max())

print("Segmented VISIBLE: ", visible_seg.shape)
print("Segmented IR: ", ir_seg.shape)
print("VISIBLE: ", vis.shape)
print("Infra-red: ", ir.shape)

#################################################
##			       Mask Generation 		       ##
#################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("############## Using GPU/CPU: {} ##############".format(device))

visible_seg = 255 - visible_seg 
ir_seg = 255 - ir_seg 

#################################################
##			     Mask Registration 		       ##
#################################################
beg = 0
end = len(visible_seg.shape[0])

visible_seg_norm =  parseImages(visible_seg, end)
visible_seg_norm = visible_seg_norm[beg:end, ...]
visible_seg_norm = visible_seg_norm.to(device)

ir_seg_norm = parseImages(ir_seg, end)
ir_seg_norm = ir_seg_norm[beg:end, ...]
ir_seg_norm = ir_seg_norm.to(device)

print("ir_seg_norm: ", ir_seg_norm.shape)
print("visible_seg_norm: ", visible_seg_norm.shape)

print("############# Loading Model      #######################")
model = AffineNet().to(device)
model = model.to(device)

model.load_state_dict(torch.load(model_name))
model.eval()

#################################################
##			  Affine Interpolation 		       ##
#################################################
wrap, flow = model(ir_seg_norm.cpu())
print("Flow shape: ", flow.shape)
print("Wrap shape: ", wrap.shape)

# Reshaping data
vis = torch.from_numpy(vis).float()
vis = vis[None, ...]
vis = vis.permute(1, 0, 2, 3)
vis_norm = vis / 255

ir = torch.from_numpy(ir).float()
ir = ir[None, ...]
ir = ir.permute(1, 0, 2, 3)
ir_norm = ir / 255

assert ir_norm.shape[0] == flow.shape[0]

x = F.grid_sample(ir_norm, flow)

print("Flow.shape: ", x.shape)
print("VIS: ", vis.shape)
print("IR: ", ir.shape)

if save_samples:
    save_pic(x, vis, ir, 20, name ="flow")
    save_pic(x, vis, ir, 80, name ="flow")
    save_pic(x, vis, ir, 15, name ="flow")
    save_pic(x, vis, ir, 30, name ="flow")
    save_pic(x, vis, ir,  9, name ="flow")
    save_pic(x, vis, ir, 50, name ="flow")
    save_pic(x, vis, ir, 51, name ="flow")
    save_pic(x, vis, ir, 74, name ="flow")
    save_pic(x, vis, ir, 34, name ="flow")
    save_pic(x, vis, ir, 38, name ="flow")
    save_pic(x, vis, ir, 47, name ="flow")
    save_pic(x, vis, ir,  7, name ="flow")
    save_pic(x, vis, ir, 11, name ="flow")

#################################################
##			 Saving registred frames           ##
#################################################
#Saving
for i in tqdm(range(1, x.shape[0])):
    name = "frame{}.jpg".format(i)
    path = saving_path + "/" + "registration" + "/" + name
    io.imsave(path, x[i, 0])

#################################################
##			  Fusion of IR & RGB 		       ##
#################################################

"""
#Saving
nom_video = "SAVE_2_visible_AFI"
path_fusion = "/ivrldata1/students/imad/data/Fusion/" + nom_video

fusion_vis = vis[:50]

for img in range(x.shape[0]):
	try:
		fT = fuse_twoscale(fusion_vis[i, 0], x[i, 0])
		name = "frame_fusion{}.jpg".format(i)
		output = path_fusion + name
		io.imsave(output, fT)
	except:
		print("\t Error while doing fusion of Frame {} \n".format(name))
"""
#################################################
##			  Alpha Blending 			       ##
#################################################
#from PIL import Image

#image2 = Image.open("C:/Users/karaimer/Desktop/registration/19440ir_mask.png")
#image1 = Image.open("C:/Users/karaimer/Desktop/registration/vis/19440vis.png")


# Make sure images got an alpha channel
#image5 = image1.convert("RGBA")
#image6 = image2.convert("RGBA")

# alpha-blend the images with varying values of alpha
#alphaBlended1 = Image.blend(image5, image6, alpha=.2)
#alphaBlended2 = Image.blend(image5, image6, alpha=.4)

# Display the alpha-blended images
#alphaBlended1 = alphaBlended1.save("alphaBlended1.png")
#alphaBlended2 = alphaBlended2.save("alphaBlended2.png")

#################################################
##			  Generating Video   		       ##
#################################################
width, height = 240, 320
outputName = "SAVE_2_AFI_VIDEO"
fps = 5
fusion = dataload(path_fusion)

saveVideo(frames, fps, width, height, outputName, fourCC='DIVX')
