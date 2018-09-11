#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:18:01 2017
@author: jordandekraker
"""
import cv2
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt


#image = cv2.imread("./SmileyFace8bitGray.png",cv2.IMREAD_GRAYSCALE)
#image = image.astype(float)
#fix = [0.5,0.5,1000.0]
        
# fisheye filter
def fixeye(image,fix):
    image = image+0.1 # otherwise can lead to cropping problems in fixeye
    outsz = [32,32]
    fix = fix.astype(float)
    
    # make sure input is square matrices
    image = np.reshape(image,[np.int(np.sqrt(image.size)),np.int(np.sqrt(image.size))])
    #convert fix to coordinates
    fix[0] = np.round(fix[0]*image.shape[0])
    fix[1] = np.round(fix[1]*image.shape[1])
    
    #set to appropriate range
    if fix[0] < 0: fix[0] = 0
    if fix[1] < 0: fix[1] = 0
#    if fix[2] < 1: fix[2] = 1.0 # lower values more distorted
    if fix[0] > image.shape[0]-1: fix[0] = image.shape[0]-1
    if fix[1] > image.shape[1]-1: fix[1] = image.shape[1]-1
#    if fix[2] > 1000: fix[0] = 1000

    
    # set up fisheye parameters
    cam = np.eye(3)
    cam[0,2] = fix[0]  # define center x
    cam[1,2] = fix[1]  # define center y
    cam[0,0] = 100        # define focal length x
    cam[1,1] = 100        # define focal length y
    #run fisheye
    dst = cv2.undistort(image,cam,1)
    
#    #crop
#    dst = dst[~np.all(dst==0,axis=0)]
#    dst = dst.T
#    dst = dst[~np.all(dst==0,axis=1)]
#    dst = dst.T

    # resize and normalize
    dst = scipy.misc.imresize(dst,outsz)
    dst = np.reshape(dst,[outsz[0]*outsz[1]]) #make 1D
    dst = dst - np.mean(dst)
    dst = dst / np.std(dst)
    return dst #.astype('float32')

#dst = fixeye(image,fix)
#dst = np.reshape(dst,[32,32])
#fig, ax = plt.subplots()
#ax.imshow(dst)