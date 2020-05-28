
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

import argparse
from utils import *
from model import AffineNet


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--inputFrames1", required=True,
    help="Path to input RGB Frames")

ap.add_argument("-i", "--inputFrames2", required=True,
    help="path to input IR Frames")

ap.add_argument("-o", "--output", required=True,
    help="Output path to save model")

ap.add_argument("-n", "--name", type=str, required=True,
    help="Model name")

ap.add_argument("-n", "--epochs", type=int, default=20, required=False,
    help="Number of epochs for training")

args = vars(ap.parse_args())

#print(args)
vis_path = args['inputFrames1']
ir_path = args['inputFrames1']
saving_path = args['output']
model_name = args['model']
epochs = int(args['epochs'])

#################################################
##                 Data Loading                ##
#################################################

visible_seg = dataload(vis_path)
ir_seg = dataload(ir_path)

#################################################
##                 Mask Generation             ##
#################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("############## Using GPU/CPU: {} ##############".format(device))

visible_seg = 255 - visible_seg 
ir_seg = 255 - ir_seg 

print("############# Started Registration #######################")
mini_batch_size = 100
optimizer = optim.Adam(model.parameters(), lr=1e-4)
model.to(device)

def train(epochs, fixed_input, moving_input, mini_batch_size):
    model.train()
    sum_loss = 0
    criterion = nn.MSELoss()
    for b in tqdm(range(0, fixed_input.size(0), mini_batch_size)):
        optimizer.zero_grad()
        #print("fixed_input.size(0) : ", fixed_input.size(0))
        #print("fixed_input.size: ", fixed_input.size())
        sample = moving_input.narrow(0, b, mini_batch_size)
        #print("Sample size", sample.size())
        warp, flow = model(sample)#, fixed_input.narrow(0, b, mini_batch_size))
        
        recon_loss = criterion(warp, fixed_input.narrow(0, b, mini_batch_size))
        loss = recon_loss
        sum_loss = sum_loss + loss.item()
        
        loss.backward()
        optimizer.step()
    print(epochs, sum_loss, loss, recon_loss)

model = model.to(device)

for epoch in range(1, epochs):
	print("#####Epoch : {}".format(epoch))
	train(epoch, visible_seg_norm, ir_seg_norm, mini_batch_size)

print("############# Done Registration #######################")

print("############# Saving Model      #######################")
name_model = model_name + ".ph"
torch.save(model.state_dict(), name_model)