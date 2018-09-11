#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 14:29:24 2018

Creates TubeNet architecture for predicting new image from image+xy motion

@author: jordandekraker
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import fixeye_saccade_Arduino as fe

mmg = cv2.imread("./SmileyFace8bitGray.png",cv2.IMREAD_GRAYSCALE)
#fig, ax = plt.subplots()
#ax.imshow(image)

sensesize = 1024
motorsize = 3
NNwidth = (sensesize+motorsize)

# initialize tensorflow trainable variables
# S=sensory, H=hippocampal, M=motor
# w=weights, b=biases, a=activations
S1w = tf.Variable(tf.random_normal([NNwidth, NNwidth])) 
S1b = tf.Variable(tf.random_normal([NNwidth])) 

S2w = tf.Variable(tf.random_normal([NNwidth, NNwidth])) 
S2b = tf.Variable(tf.random_normal([NNwidth])) 

with tf.variable_scope('rnn', reuse=tf.AUTO_REUSE):
    with tf.variable_scope('basic_lstm_cell'):
        H1w = tf.get_variable('kernel',[2*NNwidth, 4*NNwidth])
        H1b = tf.get_variable('bias',[4*NNwidth])         
H1 = tf.contrib.rnn.BasicLSTMCell(NNwidth, state_is_tuple=True, reuse=True)
H1m_c = tf.Variable(tf.random_normal([1, NNwidth])) # ititial hidden layer c_state and m_state 
H1m_h = tf.Variable(tf.random_normal([1, NNwidth]))

M2w = tf.Variable(tf.random_normal([NNwidth, NNwidth])) 
M2b = tf.Variable(tf.random_normal([NNwidth])) 

M1w = tf.Variable(tf.random_normal([NNwidth, NNwidth])) 
M1b = tf.Variable(tf.random_normal([NNwidth])) 

optimizerp = tf.train.GradientDescentOptimizer(learning_rate=0.1) 
optimizerr = tf.train.GradientDescentOptimizer(learning_rate=0.1) 
Mout = tf.Variable(tf.zeros([motorsize]))
RollingAverage = tf.Variable(tf.zeros([1]),trainable=False)
e = tf.placeholder("float64", [1])


# define the model
def tubenet(e):
    nn=tf.constant([0.])
    lossp = []
    lossr = []
    for i in range(1):
        #get new S and M
        FEimg = tf.py_func(fe.fixeye,[Mout],[tf.float64])
        FEimg = tf.cast(FEimg,tf.float32)
        S = tf.reshape(tf.concat((FEimg[0,:],Mout[:]),0),[1,NNwidth])
        
        # feed new S and M forward
        S1a = tf.tanh(tf.matmul(S, S1w) + S1b) 
        S2a = tf.tanh(tf.matmul(S1a, S2w) + S2b) 
        with tf.variable_scope('rnn') as scope:
            scope.reuse_variables()
            H1a,H1m = H1(S2a, (H1m_c,H1m_h))
        M2a = tf.tanh(tf.matmul(H1a, M2w) + M2b) # linear activation function?
        M1a = tf.matmul(M2a, M1w) + M1b # linear activation function?
        
        # predictive backprop through time
        lossp.append(tf.losses.mean_squared_error(M1a, S))
        op1 = optimizerp.minimize(lossp[i])
        
        # sometimes try random (e decreases over time)
        with tf.control_dependencies([op1]):
            op2 = Mout.assign(tf.tanh(M1a)[0,sensesize:])
        
        with tf.control_dependencies([op2]):
            op3 = Mout.assign(tf.cond(tf.random_uniform([1],0,1,dtype=tf.float64)[0] < e[0], 
                 lambda: tf.random_uniform([motorsize],-0.99,0.99),
                 lambda: Mout))
            
        # M1 reinforcement for by memory change
        with tf.control_dependencies([op3]):
            Qsignal = tf.reduce_mean(tf.abs(H1m[1]-H1m_c),1) # mean mem diff
            Q = tf.reshape(tf.tanh(M1a)[0,sensesize:],[motorsize])
            Qchange = Qsignal#-RollingAverage
            Qtarget = Q + tf.multiply(Q,Qchange)
            lossr.append(tf.reduce_mean(Qtarget-Q))
            op4 = optimizerr.minimize(lossr[i])

        # assign old values new MOVE THIS TO BEFORE R OPTIMISER?
        with tf.control_dependencies([op4]):
            op5 = RollingAverage.assign(RollingAverage*0.9 + Qsignal*0.1)
            op6 = H1m_c.assign(H1m[0])
            op7 = H1m_h.assign(H1m[1])
        # ensure things were actually run
        with tf.control_dependencies([op5,op6,op7]):
            nn = nn+1.

    return S,nn,lossp,lossr,M1a
S,nn,lossp,lossr,M1a = tubenet(e)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()
#saver.restore(sess, "tmp/TubeNet_PredictiveAutoEncoder_Arduino.ckpt")

# Start training
n = 1.
for i in range(10000):
    ee = 1./((i/50) + 10)
    s,n,lp,lr,m = sess.run([S,nn,lossp,lossr,M1a],feed_dict={e:[ee]})
    
    print(lp[0],lr[0])
    plt.subplot(1,2,1)
    plt.imshow(np.reshape(s[0,:sensesize],[32,32]))    
    plt.subplot(1,2,2)
    plt.imshow(np.reshape(m[0,:sensesize],[32,32]))
    plt.show()

saver.save(sess, "tmp/TubeNet_PredictiveAutoEncoder_Arduino.ckpt")
sess.close()
