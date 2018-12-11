#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Building upon TubeNet1.0.0, we will now try to choose new fixation locations that
have not yet been well learned using reinforcement learning. For this we will create
a branch parallel to NNlayer2, and the weights of this branch will optimize for new 
choices of M fixations. We'll call this optimizerM (as opposed to optimizerS)

@author: jordandekraker
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.stats
import cv2
import numpy as np
import scipy.misc
import urllib 
import socket
from time import sleep

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
    global image
    if (M[0]==0) & (pan>25):
        pan = pan-10
        exportMotor(pan)
        M[0] = 1
        image = importStream()
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
image = importStream()

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









# define the resolution of the fixation image and possible fixation points 
# (sense and motor, respectively)
sensesz = [32,32]
motorsz = [16,16,4]
Ssz = np.prod(sensesz)
Msz = np.prod(motorsz)
NNwidth = Msz # how wide should each NNlayer be?

# initialize tensorflow trainable variables
W1 = tf.Variable(tf.random_normal([Msz+Ssz, NNwidth]))  *(2/(Msz+Ssz+NNwidth))
b1 = tf.Variable(tf.random_normal([NNwidth])) *(2/(Msz+NNwidth))

W2 = tf.Variable(tf.random_normal([NNwidth, Ssz])) 
b2 = tf.Variable(tf.random_normal([Ssz])) 

W3 = tf.Variable(tf.random_normal([Msz, Msz]))
initialb3 = makeGaussian(motorsz[0],fwhm=motorsz[0]) # lets initialize this to favour the center
initialb3 = np.reshape(initialb3,[motorsz[0]*motorsz[1]])
initialb3 = np.repeat(initialb3,motorsz[2])
initialb3 = scipy.stats.zscore(initialb3)
b3 = tf.Variable(tf.cast(initialb3,tf.float32))

optimizerS = tf.train.GradientDescentOptimizer(learning_rate=0.1) 
optimizerM = tf.train.GradientDescentOptimizer(learning_rate=0.1) 

# initialize M and S variables
initialM = np.zeros([1,Msz])
initialM[0,200] = 1.
M = tf.Variable(tf.cast(initialM,tf.float32),trainable=False)
S = tf.Variable(tf.reshape(tf.cast(fixeye(initialM,sensesz,motorsz),tf.float32),[1,Ssz]),
                trainable=False)
RollingAverage = tf.Variable(tf.zeros([1]),trainable=False)





# define the model. Everything here will occur inside the 'tensorgraph' and 
# will not be easily accessible in our environment
def tubenet():
    nn = tf.constant([0]) #counter 

    # feed S and M forward
    h1 = tf.tanh(tf.matmul(tf.concat((M,S),1), W1) + b1) 
    h2 = tf.matmul(h1, W2) + b2 # no tanh
    h3 = tf.matmul(M, W3) + b3 # no tanh
    
    # get new S to use as training signal
    newS = tf.cast(tf.py_func(fixeye,[M,sensesz,motorsz],[tf.float64]),tf.float32)
    
    # now backpropogate
    lossS = tf.square(h2 - newS)
    op1 = optimizerS.minimize(lossS)#,var_list=[h2,W2,b2,W1,b1])

    # get new M by argmax layer 3 activity, but sometimes choose randomly
    newMind = tf.cond(tf.random_uniform([1],0,1,dtype=tf.float64)[0] <= 0.1, 
             lambda: tf.random_uniform([1],0,Msz-1,dtype=tf.int64)[0],
             lambda: tf.argmax(h3[0,:]))
    newM = tf.sparse_tensor_to_dense(tf.SparseTensor([[0,newMind]],[1.0],[1,Msz]))

    # reinforce (Q-learning) layer 3 weights to return fixations that were poorly reproduced
    reinforcer = tf.reduce_mean(lossS) # reward tricky images. note this is always positive
    Msignal = reinforcer - RollingAverage # now this can be negative or positive
    Mtarget = newM + tf.multiply(newM, Msignal)
    lossM = tf.squared_difference(Mtarget, newM)
    op2 = optimizerM.minimize(lossM)#,var_list=[h3,W3,b3])
    
    # update necessary variables
    with tf.control_dependencies([op1,op2]):
        op3 = RollingAverage.assign(RollingAverage*0.99 + reinforcer*0.01)
        op4 = S.assign(newS) 
        op5 = M.assign(newM)
    with tf.control_dependencies([op3,op4,op5]):
        nn=nn+1
    return h2,S,lossS,lossM,nn
h2,S,lossS,lossM,nn = tubenet()





# Initialize tensorgraph
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()
#saver.restore(sess, "tmp/TubeNet1.0.0.ckpt")

# Start training
iterations = 100000;
meanlossS = np.empty(iterations)
meanlossM = np.empty(iterations)
retinalImage = np.empty(sensesz)
for i in range(iterations):
    oldretinalImage = retinalImage
    TubeNet_prediction,retinalImage,LossS,LossM,n = sess.run([h2,S,lossS,lossM,nn])
    meanlossS[i] = np.mean(LossS)
    meanlossM[i] = np.mean(LossM)
    print(meanlossS[i],meanlossM[i])
    plt.subplot(1,2,1)
    plt.imshow(np.reshape(TubeNet_prediction,sensesz))    
    plt.subplot(1,2,2)
    plt.imshow(np.reshape(oldretinalImage,sensesz))
    plt.show()
    if np.remainder(i,1000)==0:
        saver.save(sess, "tmp/TubeNet2.0.0.ckpt")
#sess.close()
