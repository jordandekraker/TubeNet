#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script will import webcam stream and export motor actions to Arduino. 

@author: jordandekraker
"""
import cv2
import numpy as np
import scipy
import matplotlib.pyplot as plt
import urllib 
import socket
from time import sleep

global image, S_sz, Mfix_sz, Mhead_sz, headrange
S_sz = [32,32]
Mfix_sz = [16,16,3]
Mhead_sz = [20,20] 

#define Mhead allowable range (pan 0-160; told 0-90)
headmin = [20,10]
headrange = [120,50]

# fisheye filter
def fixeye(Mfix,n):
    global image, S_sz, Mfix_sz, Mhead_sz, headrange
    
    #convert Mfix to coordinates in a given range
    Mfix = Mfix.astype(int)
    Mfix = np.reshape(Mfix,Mfix_sz)
    Mfix = np.reshape([np.where(Mfix==1)],[3])
    Mfix = Mfix/Mfix_sz # rescale each value 0:1
    Mfix[0] = np.round(Mfix[0]*(image.shape[0]-1))
    Mfix[1] = np.round(Mfix[1]*(image.shape[1]-1))
    Mfix[2] = np.round(100.**(Mfix[2] +1)) # focal length range 100-1000
    
#    #convert Mhead to coordinates in a given range
#    Mhead = Mhead.astype(int)
#    Mhead = np.reshape(Mhead,Mhead_sz)
#    Mhead = np.reshape([np.where(Mhead==1)],[3])
#    Mhead = round(Mhead*headrange +headmin)
            
    if np.remainder(n,50)==0:
        Mhead = [np.random.randint(0,headrange[0])+headmin[0],np.random.randint(0,headrange[1])+headmin[1]]
        exportMotor(Mhead)
        image = importStream()
        while np.max(image)>255 | np.min(image)<0 | np.isnan(image).any():
            image = importStream()
        
    # set up fisheye parameters
    cam = np.eye(3)
    cam[0,2] = Mfix[0]  # define center x
    cam[1,2] = Mfix[1]  # define center y
    cam[0,0] = Mfix[2]        # define focal length x
    cam[1,1] = Mfix[2]        # define focal length y
    #run fisheye
    dst = cv2.undistort(image.astype('double'),cam,1)
    
    # crop
    x = np.where(~np.all(dst==0,axis=1))[0]
    y = np.where(~np.all(dst==0,axis=0))[0]
    dst = dst[x[0]:x[-1],y[0]:y[-1]]

    # resize and normalize
    dst = scipy.misc.imresize(dst,[S_sz[0],S_sz[1]])
    dst = np.reshape(dst,np.prod(S_sz)) #make 1D
    dst = scipy.stats.zscore(dst)
    return dst+0.000001

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
            n=1
    return i
image = importStream()

def exportMotor(Mhead):
    control=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    control.connect(('192.168.1.1',2001))
    control.send(bytes([255,1,7,Mhead[0],255])) #range 0-160deg (stops responding after 160)
    control.send(bytes([255,1,8,Mhead[1],255])) #range 0-90deg (due to hardware constraint)
    control.close()
    sleep(0.1)
    return 

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
