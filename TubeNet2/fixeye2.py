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
import urllib 
import socket
from time import sleep

#sensesz = [32,32]
#motorsz = [16,16,4]        
#M = np.zeros(motorsz)
#M[2,4,2] = 1.
pan = 90
tilt = 30

# fisheye filter
def fixeye(M,sensesz,motorsz):
    M = M.astype(int)
    
    #convert M to coordinates in a given range
    M = np.reshape(M,motorsz)
    M = np.reshape([np.where(M==1)],[3])
    M = M/motorsz # [Msz,Msz,Msz/2]
    
    # motor action contingencies
    global pan
    if (M[0]==0) & (pan>25):
        pan = pan-10
        exportMotor(pan)
        M[0] = 1
    elif (M[0]==motorsz[1]-1) & (pan<155):
        pan = pan+10
        exportMotor(pan)
        M[0] = motorsz[1]-2
    image = importStream()

    M[0] = np.round(M[0]*image.shape[0])
    M[1] = np.round(M[1]*image.shape[1])
    M[2] = np.round(100.**(M[2]+1)) # range 10-1000
    
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

def importStream():
    stream=urllib.request.urlopen('http://192.168.1.1:8080/?action=stream')
    importedbytes=''.encode()
    n=0
    while n==0:
        importedbytes+=stream.read(1024)
        a = importedbytes.find(b'\xff\xd8') #0xff 0xd8
        b = importedbytes.find(b'\xff\xd9')
        if a!=-1 and b!=-1:
            jpg = importedbytes[a:b+2]
            importedbytes= importedbytes[b+2:]
            i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            n+=1
    return i+0.001

def exportMotor(pan):
    control=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    control.connect(('192.168.1.1',2001))
    control.send(bytes([255,1,7,pan,255]))#range 0-160deg (stops responding after 160)
#    sleep(0.1)
    control.close()
    return pan

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
