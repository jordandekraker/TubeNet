#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 12:02:34 2019

Simplest TubeNet architecture (2 fully connected layers)

@author: jordandekraker
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.stats
import cv2

Ssz = [32,32] # S input layer
Msz = [16,16,4] # M input layer
sz = np.prod(Ssz)+np.prod(Msz) # total input size

############################### define TubeNet ###############################

X = tf.placeholder(tf.float64, [None, sz])
Y = tf.placeholder(tf.float64, [None, sz])

# Architecture
l1 = tf.layers.dense(inputs=X,units=sz,activation=tf.nn.tanh)
l2 = tf.layers.dense(inputs=l1,units=sz/2,activation=tf.nn.tanh)
lout = tf.layers.dense(inputs=l2,units=sz,activation=None)

# Training
loss = tf.abs(Y-lout)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss)

# Begin session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()
#saver.restore(sess, "tmp/TubeNet1.0_10000.ckpt")

############################ Define PhysicsEngine ############################

smiley = cv2.imread("misc/SmileyFace8bitGray.png",cv2.IMREAD_GRAYSCALE)
smiley = smiley.astype(float)+0.01 

def fixeye(M):
    #convert M to coordinates in a given range
    M = np.reshape(M,Msz)
    m = np.where(M==1)
    M = np.reshape(m,[3])
    M = M/Msz # rescale 0:1
    Mcost = np.sum((M-0.5)**2) # used to penalize too much motion
    M[0] = np.round(M[0]*smiley.shape[0])
    M[1] = np.round(M[1]*smiley.shape[1])
    M[2] = np.round(10.**(M[2]+1)) # range 10-10000
    # set up fisheye parameters
    cam = np.eye(3)
    cam[0,2] = M[0]  # define center x
    cam[1,2] = M[1]  # define center y
    cam[0,0] = M[2]        # define focal length x
    cam[1,1] = M[2]        # define focal length y
    #run fisheye
    dst = cv2.undistort(smiley,cam,1)
    # crop
    x = np.where(~np.all(dst==0,axis=1))[0]
    y = np.where(~np.all(dst==0,axis=0))[0]
    dst = dst[x[0]:x[-1],y[0]:y[-1]]
    # resize and normalize
    dst = scipy.misc.imresize(dst,Ssz)
    dst = np.reshape(dst,np.prod(Ssz)) #make 1D
    dst = scipy.stats.zscore(dst)
    dst = np.reshape(dst,[1,np.prod(Ssz)])
    return dst, Mcost

#################################### live ####################################

# initial feedforward
M = np.zeros([1,np.prod(Msz)])
M[0,0] = 1
S = np.zeros([1,np.prod(Ssz)])+0.001
feed_dict={X: np.concatenate((S,M),1)}
SMnew = sess.run([lout],feed_dict=feed_dict)[0]

# iterate
l = np.zeros([100000,sz]) # log the loss over time
for iters in range(100000):
    
    # parse outputs from previous iter
    Snew = np.zeros([1,np.prod(Ssz)])
    Snew[0,:] = SMnew[0,:np.prod(Ssz)]
    Mnew = np.zeros([1,np.prod(Msz)])
    if np.random.rand(1) > 1/(iters/50 +10): # sometimes take a random action
        Mnew[0,np.argmax(SMnew[0,np.prod(Ssz):])] = 1
    else:
        Mnew[0,np.random.randint(np.prod(Msz))] = 1
    
    # generate targets
    Starget,Mcost = fixeye(M)
    R = np.mean((Starget-Snew)**2) - Mcost # reinforcement signal (scalar)
    Mtarget = M+M*R

    # train current iter; feed forward for next iter
    feed_dict={X: np.concatenate((S,M),1), 
               Y: np.concatenate((Starget,Mtarget),1)}
    SMnew, l[iters,:], t = sess.run([lout, loss, train_op],feed_dict=feed_dict)
    S = Starget
    M = Mnew
    
    # benchmark
    if np.remainder(iters,10)==0:
        Sloss = np.mean(l[iters,:np.prod(Ssz)])
        Mloss = np.mean(l[iters,np.prod(Ssz):])
        print('Sloss: '+str(Sloss) +' Mloss: ' +str(Mloss))
        plt.subplot(1,2,1)
        plt.imshow(np.reshape(Starget,Ssz),cmap='gray')    
        plt.subplot(1,2,2)
        plt.imshow(np.reshape(Snew,Ssz),cmap='gray')
        plt.show()
    if np.remainder(iters,10000)==0:
        saver.save(sess, 'tmp/TubeNet1.0_iter'+str(iters)+'.ckpt')
        
sess.close()
l = l[~np.all(l==0,1)]
plt.scatter(range(l[:,0].size),np.mean(l[:,:np.prod(Ssz)],1),marker='.')
plt.scatter(range(l[:,0].size),np.mean(l[:,np.prod(Ssz):],1),marker='.')
