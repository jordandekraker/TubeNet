#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 14:29:24 2018

Creates TubeNet architecture for predicting new image from image+xy motion

@author: jordandekraker
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import fixeye2 as fe

sensesz = [32,32]
motorsz = [16,16,4]
Ssz = np.prod(sensesz)
Msz = np.prod(motorsz)
NNwidth = (Msz)

# initialize tensorflow trainable variables
# S=sensory, H=hippocampal, M=motor
# w=weights, b=biases, a=activations
S1w = tf.Variable(tf.random_normal([NNwidth, NNwidth])) 
S1b = tf.Variable(tf.random_normal([NNwidth])) 

with tf.variable_scope('rnn', reuse=tf.AUTO_REUSE):
    with tf.variable_scope('basic_lstm_cell'):
        S2w = tf.get_variable('kernel',[2*NNwidth, 4*NNwidth])
        S2b = tf.get_variable('bias',[4*NNwidth])         
S2 = tf.contrib.rnn.BasicLSTMCell(NNwidth, state_is_tuple=True, reuse=True)
S2m_c = tf.Variable(tf.random_normal([1, NNwidth])) # ititial hidden layer c_state and m_state 
S2m_h = tf.Variable(tf.random_normal([1, NNwidth]))

S3w = tf.Variable(tf.random_normal([NNwidth, Ssz])) 
S3b = tf.Variable(tf.random_normal([Ssz])) 

M2w = tf.Variable(tf.random_normal([NNwidth, NNwidth])) 
M2b = tf.Variable(tf.random_normal([NNwidth])) 

M1w = tf.Variable(tf.random_normal([NNwidth, Msz])) 
M1b = tf.Variable(tf.random_normal([Msz])) 

optimizerp = tf.train.GradientDescentOptimizer(learning_rate=0.01) 
optimizerr = tf.train.GradientDescentOptimizer(learning_rate=0.01) 
initialM = np.zeros(motorsz)
loc = (np.asarray(motorsz)/2).astype(int)
initialM[loc[0],loc[1],loc[2]] = 1.
initialM = np.reshape(initialM,[1,Msz])
M = tf.Variable(tf.cast(initialM,tf.float32),trainable=False)
S = tf.Variable(tf.reshape(tf.cast(fe.fixeye(initialM,sensesz,motorsz),tf.float32),[1,Ssz]),trainable=False)
RollingAverage = tf.Variable(tf.zeros([1]),trainable=False)

e = tf.placeholder("float64", [1])


# define the model
def tubenet(e):
    nn=tf.constant([0.])
    lossp = []
    lossr = []
    for i in range(1):
        
        # feed new S and M forward
        S1a = tf.tanh(tf.matmul(M, S1w) + S1b) 
        with tf.variable_scope('rnn') as scope:
            scope.reuse_variables()
            S2a,S2m = S2(S1a, (S2m_c,S2m_h))
        S3a = tf.matmul(S1a, S3w) + S3b # linear activation function?
        M2a = tf.tanh(tf.matmul(S2m[1], M2w) + M2b) # linear activation function?
        M1a = tf.matmul(M2a, M1w) + M1b # linear activation function?
        
        # get new M, sometimes try random (e decreases over time)
        r = tf.cond(tf.random_uniform([1],0,1,dtype=tf.float64)[0] < 1, 
             lambda: tf.random_uniform([1],0,Msz,dtype=tf.int64)[0],
             lambda: tf.argmax(M1a[0,:]))
        op1 = M.assign(tf.sparse_tensor_to_dense(tf.SparseTensor([[0,r]],[1.0],[1,Msz])))
        
        # get new S
        with tf.control_dependencies([op1]):
            op2 = S.assign(tf.cast(tf.py_func(fe.fixeye,[M,sensesz,motorsz],[tf.float64]),tf.float32))
        
        # optimize backprop by prediction through time
        with tf.control_dependencies([op2]):
            lossp.append(tf.losses.mean_squared_error(S3a, S))
            op3 = optimizerp.minimize(lossp[i],var_list=[S1w,S1b,S2w,S2b,S3w,S3b])
        
        # optimize reinforcement by memory change
        with tf.control_dependencies([op3]):
            Qsignal = tf.reduce_mean(tf.abs(S2m[1]-S2m_c),1) # mean mem diff
            Q = M
            Qchange = Qsignal 
            Qtarget = Q + tf.multiply(Q,Qchange)
            lossr.append(tf.reduce_mean(Qtarget-Q))
            op4 = optimizerr.minimize(lossr[i],var_list=[Q,M1w,M1b,M2w,M2b])

        # assign old values new MOVE THIS TO BEFORE R OPTIMISER?
        with tf.control_dependencies([op4]):
            op5 = RollingAverage.assign(RollingAverage*0.9 + Qsignal[0]*0.1)
            op6 = S2m_c.assign(S2m[0])
            op7 = S2m_h.assign(S2m[1])
        # ensure things were actually run
        with tf.control_dependencies([op5,op6,op7]):
            nn = nn+1.

    return S,nn,lossp,lossr,S3a
S,nn,lossp,lossr,S3a = tubenet(e)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()
#saver.restore(sess, "tmp/TubeNet2.0.ckpt")

# Start training
n = 1.
for i in range(10000):
    ee = 1./((i/50) + 10)
    s,n,lp,lr,p = sess.run([S,nn,lossp,lossr,S3a],feed_dict={e:[ee]})
    
    print(lp[0],lr[0])
    plt.subplot(1,2,1)
    plt.imshow(np.reshape(s[0,:Ssz],[32,32]))    
    plt.subplot(1,2,2)
    plt.imshow(np.reshape(p[0,:Ssz],[32,32]))
    plt.show()

#saver.save(sess, "tmp/TubeNet2.0.ckpt")
sess.close()
