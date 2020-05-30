import os, sys
import numpy as np
import keras.layers
from keras import backend as K
import cv2
from tqdm import tqdm
from skimage.transform import resize
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from skimage import io
from skimage import color
from imageio import imread, imwrite

from utils import *
from skimage.transform import resize

#Fusion Model
sys.path.append('./fusion/')

from filters import *
import vggfusiongpu as fu
from torchvision.models.vgg import vgg19


model = vgg19(True).cuda().eval()


print("\t################# Loading Datasets #################")
path_vis2 = "./SAVE_2_visible_AFI"



def fuse_twoscale(vis, ir, **kwargs):
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

#Saving
nom_video = "SAVE_2_visible_AFI"
path_registration = "./Registred/SAVE_2_visible_AFI/" + nom_video

print("\t################# Fusion of IR & RGB ##############")
#Saving
nom_video = "SAVE_2_visible_AFI"
path_fusion = "./Fusion/" + nom_video


#################################################
##			  Fusion of IR & RGB 		       ##
#################################################
for i in tqdm(range(1, 51)):
    name = "frame"+str(i)+".jpg"
    img_reg = path_registration + "/" + name
    img_path_vis = path_vis2 + "/" + name
    
    #print(type(img_path_vis))
    print(img_reg)
    ir_sample = color.rgb2gray(imread(img_reg))
    vis_sample = color.rgb2gray(imread(img_path_vis))
    
    #ir_sample = ir_sample.astype(np.int64)
    #vis_sample = vis_sample.astype(np.int64)
    
    #print("VIS_sample", type(vis_sample))
    #print("IR_sample", type(ir_sample))
    
    print("VIS_sample.shape: ", vis_sample.shape)
    print("IR_sample.shape: ",ir_sample.shape)
    
    #try:
    ir_sample = resize(ir_sample, (256, 256))
    ir_sample = ir_sample * 256
    
    ir_sample = ir_sample.astype(np.int64)
    vis_sample = vis_sample.astype(np.int64)
    
    fT = fuse_twoscale(vis_sample, ir_sample)
    fusion_name = "frame{}.jpg".format(i)
    print("fusion name: ", )
    output = path_fusion + "/" + fusion_name
    io.imsave(output, fT)
    