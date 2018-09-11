#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script will create 'retinal images' based on the motor fixation location,
 M. In this case the image itself is simply "misc/SmileyFace8bitGray.png". 
 sensesz and motorsz represent the resolution of the output image and the 
 possible fixation locations

@author: jordandekraker
"""
import cv2
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

#sensesz = [32,32]
#motorsz = [16,16,4]        
#M = np.zeros(motorsz)
#M[2,4,2] = 1.

# fisheye filter
def fixeye(M,sensesz,motorsz):
    image = cv2.imread("misc/SmileyFace8bitGray.png",cv2.IMREAD_GRAYSCALE)
    image = image.astype(float)+0.1 # otherwise can lead to cropping problems in Meye
    M = M.astype(int)
    
    # make sure input is square matrices
    image = np.reshape(image,[np.int(np.sqrt(image.size)),np.int(np.sqrt(image.size))])
    
    #convert M to coordinates in a given range
    M = np.reshape(M,motorsz)
    M = np.reshape([np.where(M==1)],[3])
    M = M/motorsz # [Msz,Msz,Msz/2]
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
    
    # crop
    x = np.where(~np.all(dst==0,axis=1))[0]
    y = np.where(~np.all(dst==0,axis=0))[0]
    dst = dst[x[0]:x[-1],y[0]:y[-1]]

    # resize and normalize
    dst = scipy.misc.imresize(dst,sensesz)
    dst = np.reshape(dst,[sensesz[0]*sensesz[1]]) #make 1D
    dst = dst - np.mean(dst)
    dst = dst / np.std(dst)
    return dst.astype('double')

def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

#dst = fixeye(M,sensesz,motorsz)
#dst = np.reshape(dst,sensesz)
#fig, ax = plt.subplots()
#ax.imshow(dst)