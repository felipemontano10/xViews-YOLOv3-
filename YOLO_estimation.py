#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 15:52:29 2020

@author: felipemontanocampos
"""

import os
path = "/Users/felipemontanocampos/Dropbox/Third Semester/Deep Learning/Final Project"
os.chdir(path)







#object detection


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image, ImageDraw
import tensorflow as tf
import io
import glob
from tqdm import tqdm
import numpy as np
import logging
import argparse
import os
import json
import csv
import skimage.filters as filters

##########################
##### Training Set #######
##########################



#Get the labels of the images 

def get_labels(fname="xView_train.geojson"):
    
    with open(fname) as f:
        data = json.load(f)
    
    coords = np.zeros((len(data['features']),4))
    chips = np.zeros((len(data['features'])),dtype="object")
    classes = np.zeros((len(data['features'])))
    
    for i in range(len(data['features'])):
        if data['features'][i]['properties']['bounds_imcoords'] != []:
            b_id = data['features'][i]['properties']['image_id']
            val = np.array([int(num) for num in data['features'][i]['properties']['bounds_imcoords'].split(",")])
            chips[i] = b_id
            classes[i] = data['features'][i]['properties']['type_id']
            coords[i] = val
        else:
            chips[i] = 'None'
            
    return coords, chips, classes

#Let's get the coordintates, chip images and the classes
    
coords, chips, classes = get_labels()

#Define the chipping image 

def chip_image(img,coords,classes,shape=(300,300)):
    """
    Chip an image and get relative coordinates and classes.  Bounding boxes that pass into
        multiple chips are clipped: each portion that is in a chip is labeled. For example,
        half a building will be labeled if it is cut off in a chip. If there are no boxes,
        the boxes array will be [[0,0,0,0]] and classes [0].
        Note: This chip_image method is only tested on xView data-- there are some image manipulations that can mess up different images.
    Args:
        img: the image to be chipped in array format
        coords: an (N,4) array of bounding box coordinates for that image
        classes: an (N,1) array of classes for each bounding box
        shape: an (W,H) tuple indicating width and height of chips
    Output:
        An image array of shape (M,W,H,C), where M is the number of chips,
        W and H are the dimensions of the image, and C is the number of color
        channels.  Also returns boxes and classes dictionaries for each corresponding chip.
    """
    height,width,_ = img.shape
    wn,hn = shape
    
    w_num,h_num = (int(width/wn),int(height/hn))
    images = np.zeros((w_num*h_num,hn,wn,3))
    total_boxes = {}
    total_classes = {}
    
    k = 0
    for i in range(w_num):
        for j in range(h_num):
            x = np.logical_or( np.logical_and((coords[:,0]<((i+1)*wn)),(coords[:,0]>(i*wn))),
                               np.logical_and((coords[:,2]<((i+1)*wn)),(coords[:,2]>(i*wn))))
            out = coords[x]
            y = np.logical_or( np.logical_and((out[:,1]<((j+1)*hn)),(out[:,1]>(j*hn))),
                               np.logical_and((out[:,3]<((j+1)*hn)),(out[:,3]>(j*hn))))
            outn = out[y]
            out = np.transpose(np.vstack((np.clip(outn[:,0]-(wn*i),0,wn),
                                          np.clip(outn[:,1]-(hn*j),0,hn),
                                          np.clip(outn[:,2]-(wn*i),0,wn),
                                          np.clip(outn[:,3]-(hn*j),0,hn))))
            box_classes = classes[x][y]
        
            if out.shape[0] != 0:
                total_boxes[k] = out
                total_classes[k] = box_classes
            else:
                total_boxes[k] = np.array([[0,0,0,0]])
                total_classes[k] = np.array([0])
            
            chip = img[hn*j:hn*(j+1),wn*i:wn*(i+1),:3]
            images[k]=chip
            
            k = k + 1
    
    return images.astype(np.uint8),total_boxes,total_classes

#Auxiliar function to get the images 
def get_image(fname):    
    """
    Get an image from a filepath in ndarray format
    """
    return np.array(Image.open(fname))

#Work with a reduced number of 10 images
import glob
image_folder= 'imagedata/'
fnames = glob.glob(image_folder + "*.tif")
fnames.sort()

#Reduced Datasets
new_chips=[]
new_classes=[]
new_coords=[]


for fname in fnames:
            
            name = fname.split("/")[-1]
            arr = get_image(fname)
            im,box,classes_final = chip_image(arr,coords[chips==name],classes[chips==name])
            for idx, image in enumerate(im):

                
                if not box[idx].any():
                    continue
                
                Image.fromarray(image).save('process/img_%s_%s.png'%(name,idx))
                new_chips.extend(['img_%s_%s.png'%(name,idx)]*len(box[idx]))
                new_classes.extend(classes_final[idx])
                new_coords.extend(box[idx])



new_coords=np.array(new_coords)
new_classes= np.array(new_classes)
new_chips= np.array(new_chips)

new_coords=new_coords.T

#Get the training 


train = pd.DataFrame({'image': new_chips, 'xmin':new_coords[0].astype(int),'ymin':new_coords[1].astype(int),\
                      'xmax':new_coords[2].astype(int),\
                        'ymax':new_coords[3].astype(int) , 'label':new_classes})


train.to_csv(r"/Users/felipemontanocampos/Dropbox/TrainYourOwnYOLO/Data/Source_Images/Training_Images/vott-csv-export/Annotations-export.csv")





######################
##### Test Set #######
######################

import glob
image_folder= 'Testdata/'
fnames = glob.glob(image_folder + "*.tif")
fnames.sort()

new_chips=[]
new_classes=[]
new_coords=[]

for fname in fnames:
            
            name = fname.split("/")[-1]
            arr = get_image(fname)
            im,box,classes_final = chip_image(arr,coords[chips==name],classes[chips==name])
            for idx, image in enumerate(im):

                
                if not box[idx].any():
                    continue

                Image.fromarray(image.astype(np.uint8)).save('testprocess/img_%s_%s.png'%(name,idx))
                new_chips.extend(['img_%s_%s.png'%(name,idx)]*len(box[idx]))
                new_classes.extend(classes_final[idx])
                new_coords.extend(box[idx])


new_coords=np.array(new_coords)
new_classes= np.array(new_classes)
new_chips= np.array(new_chips)

new_coords=new_coords.T





test = pd.DataFrame({'image': new_chips, 'xmin':new_coords[0].astype(int),'ymin':new_coords[1].astype(int),\
                      'xmax':new_coords[2].astype(int),\
                        'ymax':new_coords[3].astype(int) , 'label':new_classes})



##########################
####Train the model ######
##########################


##Create data in YOLO format

path = "/Users/felipemontanocampos/Dropbox/TrainYourOwnYOLO/1_Image_Annotation"
os.chdir(path)
runfile("Convert_to_YOLO_format.py")

##Create and Convert Pre-Trained Weights

path = "/Users/felipemontanocampos/Dropbox/TrainYourOwnYOLO/2_Training"
os.chdir(path)

## Train the model 

runfile("Download_and_Convert_YOLO_weights.py")
runfile("Train_YOLO.py")

## Test my model 

path = "/Users/felipemontanocampos/Dropbox/TrainYourOwnYOLO/3_Inference"
os.chdir(path)
runfile("Detector.py")













