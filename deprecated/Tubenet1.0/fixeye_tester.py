#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 17:06:39 2018

@author: jordandekraker
"""

import numpy as np
import fixeye_saccade_smiley as fe
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

MNISTimg, MNISTcat = mnist.train.next_batch(1)

image = np.reshape(MNISTimg,[np.int(np.sqrt(MNISTimg.size)),np.int(np.sqrt(MNISTimg.size))])
fig, ax = plt.subplots()
ax.imshow(image)

fix = np.zeros([1,100])
fix[0,55] = 1.0
image = fe.fixeye(MNISTimg,fix[0,:])
image = np.reshape(image,[np.int(np.sqrt(image.size)),np.int(np.sqrt(image.size))])
fig, ax = plt.subplots()
ax.imshow(image)

fix = np.zeros([100])
fix[55] = 1
image = fe.fixeye(MNISTimg,fix)
image = np.reshape(image,[np.int(np.sqrt(image.size)),np.int(np.sqrt(image.size))])
fig, ax = plt.subplots()
ax.imshow(image)

fix = np.zeros([1,100])
fix[0,55] = 1
image = fe.fixeye(MNISTimg,fix)
image = np.reshape(image,[np.int(np.sqrt(image.size)),np.int(np.sqrt(image.size))])
fig, ax = plt.subplots()
ax.imshow(image)

