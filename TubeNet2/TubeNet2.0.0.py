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
import fixeye2 as fe
import scipy.stats

S_sz = fe.S_sz
Mfix_sz = fe.Mfix_sz
Mhead_sz = fe.Mhead_sz
Ssz = np.prod(S_sz)
Mfix_n = np.prod(Mfix_sz)
NNwidth = Mfix_n # how wide should each NNlayer be?

# initialize tensorflow trainable variables
W1 = tf.Variable(tf.random_normal([Mfix_n+Ssz, NNwidth]))  *(2/(Mfix_n+Ssz+NNwidth))
b1 = tf.Variable(tf.random_normal([NNwidth])) *(2/(Mfix_n+NNwidth))

W2 = tf.Variable(tf.random_normal([NNwidth, Ssz])) 
b2 = tf.Variable(tf.random_normal([Ssz])) 

W3 = tf.Variable(tf.random_normal([Ssz, NNwidth])) 
b3 = tf.Variable(tf.random_normal([NNwidth])) 

W4 = tf.Variable(tf.random_normal([NNwidth, Mfix_n]))
b4 = tf.Variable(tf.random_normal([Mfix_n])) 
#initialb4 = fe.makeGaussian(Mfix_sz[0],fwhm=Mfix_sz[0]) # lets initialize this to favour the center
#initialb4 = np.reshape(initialb4,[Mfix_sz[0]*Mfix_sz[1]])
#initialb4 = np.repeat(initialb4,Mfix_sz[2])
#initialb4 = scipy.stats.zscore(initialb4)
#b4 = tf.Variable(tf.cast(initialb4,tf.float32))

init = tf.train.GradientDescentOptimizer(learning_rate=0.1) 
optimizerS = tf.contrib.estimator.clip_gradients_by_norm(init, clip_norm=5.0)
init = tf.train.GradientDescentOptimizer(learning_rate=0.1) 
optimizerMfix = tf.contrib.estimator.clip_gradients_by_norm(init, clip_norm=5.0)

# initialize Mfix and S variables
initialMfix = np.zeros([1,Mfix_n])
initialMfix[0,200] = 1.
Mfix = tf.Variable(tf.cast(initialMfix,tf.float32),trainable=False)
S = tf.Variable(tf.reshape(tf.cast(fe.fixeye(initialMfix,1),tf.float32),[1,Ssz]),
                trainable=False)
RollingAverage = tf.Variable(tf.zeros([1]),trainable=False)
n = tf.placeholder(tf.int32)





# define the model. Everything here will occur inside the 'tensorgraph' and 
# will not be easily accessible in our environment

# feed S and Mfix forward
h1 = tf.tanh(tf.matmul(tf.concat((Mfix,S),1), W1) + b1) 
h2 = tf.matmul(h1, W2) + b2 # no tanh
h3 = tf.tanh(tf.matmul(h2, W3) + b3) 
h4 = tf.matmul(Mfix, W4) + b4 # no tanh; stop Mfix optim

# get new S to use as training signal
newS = tf.cast(tf.py_func(fe.fixeye,[Mfix,n],[tf.float64]),tf.float32)

# now backpropogate
lossS = tf.square(newS - h2)
op1 = optimizerS.minimize(lossS)#,var_list=[h2,W2,b2,h1,W1,b1])

# get new Mfix by argmax layer 3 activity, but sometimes choose randomly
newMfixind = tf.cond(tf.random_uniform([1],0,1,dtype=tf.float64)[0] <= 0.1, 
         lambda: tf.random_uniform([1],0,Mfix_n-1,dtype=tf.int64)[0],
         lambda: tf.argmax(h4[0,:]))
newMfix = tf.sparse_tensor_to_dense(tf.SparseTensor([[0,newMfixind]],[1.0],[1,Mfix_n]))

# reinforce (Q-learning) layer 3 weights to return fixations that were poorly reproduced
reinforcer = tf.reduce_mean(lossS) # reward tricky images. note this is always positive
Mfixsignal = reinforcer - RollingAverage # now this can be negative or positive
Mfixtarget = newMfix + tf.multiply(newMfix, Mfixsignal)
lossMfix = tf.square(newMfix - Mfixtarget)
op2 = optimizerMfix.minimize(lossMfix)#,var_list=[h4,W4,b4])

# update necessary variables
with tf.control_dependencies([op1,op2]):
    op3 = RollingAverage.assign(RollingAverage*0.99 + reinforcer*0.01)
    op4 = S.assign(newS) 
    op5 = Mfix.assign(newMfix)
with tf.control_dependencies([op3,op4,op5]):
    nn=n+1






# Initialize tensorgraph
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()
#saver.restore(sess, "tmp/TubeNet1.0.0.ckpt")

# Start training
iterations = 100000;
meanlossS = np.empty(iterations)
meanlossMfix = np.empty(iterations)
retinalImage = np.empty(S_sz)
for i in range(iterations):
    oldretinalImage = retinalImage
    TubeNet_prediction,retinalImage,LossS,LossMfix,nnn = sess.run([h2,S,lossS,lossMfix,nn],
                                                                feed_dict={n:i})
    meanlossS[i] = np.mean(LossS)
    meanlossMfix[i] = np.mean(LossMfix)
    if np.any(
        [[meanlossS[i]>100, meanlossS[i]<0.0, np.isnan(meanlossS[i])],
        [meanlossMfix[i]>100, meanlossMfix[i]<0.0, np.isnan(meanlossMfix[i])],
        [TubeNet_prediction.any()>100, TubeNet_prediction.any()<0.0, np.isnan(TubeNet_prediction.any())]]):
        break
#    if np.remainder(i,20)==0:
    print(meanlossS[i],meanlossMfix[i],nnn)
    plt.subplot(1,2,1)
    plt.imshow(np.reshape(TubeNet_prediction,S_sz),cmap='gray')    
    plt.subplot(1,2,2)
    plt.imshow(np.reshape(retinalImage,S_sz),cmap='gray')
    plt.show()
#    if np.remainder(i,50)==0:
#        fe.exportMotor(np.random.randint(0,160),np.random.randint(0,90))
#    if np.remainder(i,1000)==0:
#        saver.save(sess, "tmp/TubeNet2.0.0.ckpt")
#sess.close()
