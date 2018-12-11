#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this script, we create a simple NN that will reproduce a given image based 
 on where the fixeye filter is looking (M) alone.

@author: jordandekraker
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import fixeye1 as fe

# define the resolution of the fixation image and possible fixation points 
# (sense and motor, respectively)
sensesz = [32,32]
motorsz = [8,8,4]
Ssz = np.prod(sensesz)
Msz = np.prod(motorsz)
NNwidth = Msz # how wide should each NNlayer be?

# initialize tensorflow trainable variables
# w=weights, b=biases, a=activations
NNlayer1w = tf.Variable(tf.random_normal([Msz, NNwidth])) 
NNlayer1b = tf.Variable(tf.random_normal([NNwidth])) 

NNlayer2w = tf.Variable(tf.random_normal([NNwidth, Ssz])) 
NNlayer2b = tf.Variable(tf.random_normal([Ssz])) 

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001) 

# initialize M and S variables
initialM = np.zeros([1,Msz])
initialM[0,0] = 1.
M = tf.Variable(tf.cast(initialM,tf.float32),trainable=False)
S = tf.Variable(tf.reshape(tf.cast(fe.fixeye(initialM,sensesz,motorsz),tf.float32),[1,Ssz]),
                trainable=False)

# define the model. Everything here will occur inside the 'tensorgraph' and 
# will not be easily accessible in our environment
def tubenet():
    nn = tf.constant([0]) #counter 

    # feed S and M forward
    NNlayer1a = tf.tanh(tf.matmul(tf.concat((M),1), NNlayer1w) + NNlayer1b) 
    NNlayer2a = tf.matmul(NNlayer1a, NNlayer2w) + NNlayer2b # no tanh
    
    # get new S to use as training signal
    newS = tf.cast(tf.py_func(fe.fixeye,[M,sensesz,motorsz],[tf.float64]),tf.float32)
    
    # now backpropogate
    loss = tf.square(NNlayer2a - newS)
    op1 = optimizer.minimize(loss)

    # get new random M and embed into an empty matrix
    newM = tf.sparse_tensor_to_dense(tf.SparseTensor(
            [[0,tf.random_uniform([1],0,Msz-1,dtype=tf.int64)[0]]],[1.0],[1,Msz]))
    
    # update necessary variables
    oldS = S
    op2 = S.assign(newS) 
    op3 = M.assign(newM) 
    # ensure operations were run! (tensorgraph gets lazy if it can)
    with tf.control_dependencies([op1,op2,op3]):
        nn=nn+1
    return NNlayer2a,oldS,loss,nn
NNlayer2a,S,loss,nn = tubenet()

# Initialize tensorgraph
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
#saver = tf.train.Saver()
#saver.restore(sess, "tmp/TubeNet1.0.0.ckpt")

# Start training
iterations = 10000;
meanloss = np.empty(iterations)
retinalImage = np.empty(sensesz)
for i in range(iterations):
    oldretinalImage = retinalImage
    TubeNet_prediction,retinalImage,Loss,n = sess.run([NNlayer2a,S,loss,nn])
    meanloss[i] = np.mean(Loss)
    print(meanloss[i])
    plt.subplot(1,2,1)
    plt.imshow(np.reshape(TubeNet_prediction,sensesz))    
    plt.subplot(1,2,2)
    plt.imshow(np.reshape(oldretinalImage,sensesz))
    plt.show()

#saver.save(sess, "tmp/TubeNet1.0.0.ckpt")
#sess.close()
