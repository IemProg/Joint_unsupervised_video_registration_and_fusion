#!/usr/bin/python

import cv2
import numpy as np
import glob, os
from tqdm import tqdm
 
img_array = []
path = input("Enter Path to Frame: ")
for filename in tqdm((sorted(glob.glob(path+"/*.jpg"), key = os.path.getmtime))):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    print(size)
    img_array.append(img)
 
 
out = cv2.VideoWriter('fusion_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()