import cv2
import numpy as np
import glob
 
nom_video = "new_imad_2"
path_fusion = "/Users/user/Desktop/UPMC/EPFL/Research/testing/dataset/Fusion/" + nom_video

img_array = []
for filename in glob.glob('/Users/user/Desktop/UPMC/EPFL/Research/testing/dataset/Fusion/new_imad_2/*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('fusion_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()