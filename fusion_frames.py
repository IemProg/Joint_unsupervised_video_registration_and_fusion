#!/usr/bin/env python3
# coding: utf-8

import glob
import os

import numpy as np
from imageio import imread, imwrite
from skimage.measure import compare_ssim

import torch
import torch.nn
from torchvision.models.vgg import vgg19
import vggfusiongpu as fu

import cv2

from skimage import morphology
from skimage import io
from skimage import color 

from filters import *

from tqdm import tqdm

model = vgg19(True).cuda().eval()

def fuse_twoscale(V, I, **kwargs):
    with_exp = kwargs.get('with_exp', True)
    kernel = kwargs.get('kernel', 45)
    r1 = kwargs.get('r1', 45)
    eps1 = kwargs.get('eps1', 0.01)
    r2 = kwargs.get('r2', 7)
    eps2 = kwargs.get('eps2', 10e-6)
    layer = kwargs.get('layer', 2)
    
    k = (kernel, kernel)
    
    # Decomposition
    V = vis / 255.
    Bv = cv2.blur(V, k) 
    Dv = V - Bv

    I = ir / 255.
    Bi = cv2.blur(I, k)
    Di = I - Bi
    
    # Base Fusion
    P1 = SalWeights([vis, ir])
    P1 = [P1[:,:,0], P1[:,:,1]]
    Wb = GuidedOptimize([V, I], P1, r1, eps1)
    fT2 = FuseWeights([Bv, Bi], Wb)

    # Detail Fusion
    DT = fu.fuse([Dv, Di], model, with_exp=with_exp, layer_number=layer)
    Wd = DT[0].cpu().numpy()
    fT1 = FuseWeights([Dv, Di], np.dstack(Wd))
    
    # Reconstruction
    fT = fT2 + fT1
    fT = np.clip(fT, 0, 1)
    fT = (fT*255).astype(np.uint8)
    
    return fT

print("--------- Importing data ----------")
# local imports.
path1 = "/users/Etu2/3801582/re/fusion/SAVE_1_visible_frames_resized"
path2 = "/users/Etu2/3801582/re/fusion/SAVE_1_ir0_frames_resized"
onlyfiles1 = [f for f in os.listdir(path1) if os.path.isfile(os.path.join(path1, f))]
onlyfiles2 = [f for f in os.listdir(path2) if os.path.isfile(os.path.join(path2, f))]

print("Working with {0} VISIBLE images".format(len(onlyfiles1)))
print("Working with {0} IR images".format(len(onlyfiles2)))

# For this partis we will be working on only 50 First frame to us the results of the Fusion
#vis_first = onlyfiles1[:1500]
#ir_first = onlyfiles2[:1500] 
print("--------- Reading VISIBLE/IR images -----------")
print("--------- Images will be saved in: Results Directory -----------")
try:
    # Create target Directory
    dirName = "results"
    os.mkdir(dirName)
    print("Directory " , dirName ,  " Created ") 
except FileExistsError:
    print("Directory " , dirName ,  " already exists")

print("------ Fusion of Images ----------")
for i in tqdm(range(1, len(onlyfiles1))):
    name = "frame"+str(i)+".jpg"
    img_path_vis = path1 + "/" + name
    img_path_ir = path2 + "/" + name
    #TODO: Images should be ranged from [0 , 255]
    
    ir = color.rgb2gray(imread(img_path_ir))
    vis = color.rgb2gray(imread(img_path_vis))

    #We have to re-size its intensities due to that I saved then normalized [0, 1]
    for x in range(ir.shape[0]):
        for y in range(ir.shape[1]):
            ir[x, y] = int(ir[x, y] * 255)
            vis[x, y] = int(vis[x, y] * 255)

    ir = ir.astype(np.int64)
    vis = vis.astype(np.int64)

    try:
        fT = fuse_twoscale(vis, ir)
        output = "results/"+name[:-4] + "_fusion.jpg"
        io.imsave(output, fT)
    except:
        print("\t Error while doing fusion of Frame {} \n".format(name))
    #print("{} was succussfully saved ! ".format(output))


print("-----------------------------------------------------------------")
print("--------- Images succussfully Fusioned ! ---------")
print("\t--------- Creating a video file ---- ")
 
img_array = []
path = os.path.abspath(os.getcwd())  + "/results"
for filename in tqdm((sorted(glob.glob(path + "/*.jpg"), key = os.path.getmtime))):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('fusion_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

print("\t--------- Created:  fusion_video.avi ----")
