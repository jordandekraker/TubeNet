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
        
#M = np.zeros([32,32,4])
#M[16,16,1] = 1.

# fisheye filter
def fixeye(M):
    image = cv2.imread("./SmileyFace8bitGray.png",cv2.IMREAD_GRAYSCALE)
    image = image.astype(float)
#    image = image+0.1 # otherwise can lead to cropping problems in Meye
    outsz = [32,32]
    M = M.astype(float)
    
    # make sure input is square matrices
    image = np.reshape(image,[np.int(np.sqrt(image.size)),np.int(np.sqrt(image.size))])
    
    #convert M to coordinates in a given range
    M = np.reshape(M,[16,16,4])
    M = np.reshape(np.asarray(np.where(M==1.)),[3])
    M = M/[16,16,2] # [Msz,Msz,Msz/2]
    M[0] = np.round(M[0]*image.shape[0])
    M[1] = np.round(M[1]*image.shape[1])
    M[2] = np.round(10.**(M[2]+1)) # range 10-1000
    
    
    # set up fisheye parameters
    cam = np.eye(3)
    cam[0,2] = M[0]  # define center x
    cam[1,2] = M[1]  # define center y
    cam[0,0] = M[2]        # define focal length x
    cam[1,1] = M[2]        # define focal length y
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

#dst = fixeye(M)
#dst = np.reshape(dst,[32,32])
#fig, ax = plt.subplots()
#ax.imshow(dst)