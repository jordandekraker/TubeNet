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
import fixeye1 as fe
import scipy.stats

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
initialb3 = fe.makeGaussian(motorsz[0],fwhm=motorsz[0]) # lets initialize this to favour the center
initialb3 = np.reshape(initialb3,[motorsz[0]*motorsz[1]])
initialb3 = np.repeat(initialb3,motorsz[2])
initialb3 = scipy.stats.zscore(initialb3)
b3 = tf.Variable(tf.cast(initialb3,tf.float32))

optimizerS = tf.train.GradientDescentOptimizer(learning_rate=0.1) 
optimizerM = tf.train.GradientDescentOptimizer(learning_rate=0.1) 

# initialize M and S variables
initialM = np.zeros([1,Msz])
initialM[0,0] = 1.
M = tf.Variable(tf.cast(initialM,tf.float32),trainable=False)
S = tf.Variable(tf.reshape(tf.cast(fe.fixeye(initialM,sensesz,motorsz),tf.float32),[1,Ssz]),
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
    newS = tf.cast(tf.py_func(fe.fixeye,[M,sensesz,motorsz],[tf.float64]),tf.float32)
    
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
#saver = tf.train.Saver()
#saver.restore(sess, "tmp/TubeNet1.0.0.ckpt")

# Start training
iterations = 10000;
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

#saver.save(sess, "tmp/TubeNet1.0.0.ckpt")
#sess.close()
