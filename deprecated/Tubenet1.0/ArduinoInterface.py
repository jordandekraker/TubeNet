#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 10:43:25 2018

@author: jordan
"""
import matplotlib.pyplot as plt
import cv2
import urllib 
import numpy as np
import socket
from time import sleep

def look():
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
            i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            n+=1
    i = cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)
    return i

def move(layer):
    control=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    control.connect(('192.168.1.1',2001))
    
    stop = b'\xff\x00\x00\x00\xff'
    forward=b'\xff\x00\x01\x00\xff'
    backward=b'\xff\x00\x02\x00\xff'
    left = b'\xff\x00\x03\x00\xff'
    right = b'\xff\x00\x04\x00\xff'
    
    pan = bytes([255,1,7,90,255]) #range 0-160deg (stops responding after 160)
    tilt =bytes([255,1,8,30,255]) #range 0-90deg (due to hardware constraint)
    control.send(pan)
    control.send(tilt)
    
    control.send(forward)
    sleep(.1) #min time to reliably respond is 0.02s
    control.send(stop);
    sleep(.1) 
    control.send(backward);
    sleep(.1) 
    control.send(stop);
    sleep(.1) 
    control.send(left);
    sleep(.1) 
    control.send(stop);
    sleep(.1) 
    control.send(right);
    sleep(.1) 
    control.send(stop);
    sleep(.1) 
    control.send(right);
    sleep(.1) 
    control.send(left);
    sleep(.1) 
    control.send(stop);
