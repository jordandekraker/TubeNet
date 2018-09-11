#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 13:06:26 2018

tests MNISTperformance on single vs dual optimizer models

@author: jordandekraker
"""

nfixes = 10
iters = 100000 #10k training images seen 10 times each
tests = 20
disp_n_iters = int(iters/10)
pltwindow = int(iters/100)
test_len = 1000 #1k test images

import scipy
import numpy as np
import tensorflow as tf
import fixeye_saccade as fe
import matplotlib.pyplot as plt

inputshape = 25
fixshape = 100
memshape = 100
outshape = 10

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
loaded = True

scripts = ['TubeNet_MNIST_S1H1M1_Qlrn-centrefix.py',
           'TubeNet_MNIST_S1H1M1_Qlrn-randfix.py',
           'TubeNet_MNIST_S1H1M1_Qlrn-MNIST.py',
           'TubeNet_MNIST_S1H1M1_Qlrn-convergence.py',
           'TubeNet_MNIST_S1H1M1_Qlrn-memdiff.py']

allaccperiter = np.zeros([iters,tests,len(scripts)])
alltestacc = np.zeros([tests,len(scripts)])
for ntests in range(tests):
    for s in range(len(scripts)):
        runfile('/kohlerlab/Jordan/deepLearning/'+scripts[s])
        allaccperiter[:,ntests,s] = accuracyperiter
        alltestacc[ntests,s] = np.mean(finalaccuracyperiter)
        print(scripts[s])

trainingplt = np.zeros([int(iters/pltwindow),len(scripts)])
meanallaccperiter = np.zeros([iters,len(scripts)])
for s in range(len(scripts)):
    print(scripts[s] + ": mean test accuracy =" + "{:.4f}".format(np.mean(alltestacc[:,s])))    
    #plot accuracy over the course of training
    meanallaccperiter[:,s] = np.mean(allaccperiter[:,:,s],1)
    for aw in range(0,iters,pltwindow):
        trainingplt[int(aw/pltwindow),s] = np.mean(meanallaccperiter[aw:aw+pltwindow,s])
        
for s in range(len(scripts)):
    plt.plot(trainingplt[:,s], label=scripts[s])
plt.xlabel("training iteration (*" + str(pltwindow) + ")")
plt.ylabel("MNIST accuracy")
plt.legend()
axes = plt.gca()
#axes.set_ylim([0.7,1.0])
plt.show()

#t,p = scipy.stats.ttest_ind(alltestacc[:,0],alltestacc[:,1])
#print(p)
        
plt.bar(range(s+1),np.mean(alltestacc,0),yerr=np.std(alltestacc,0),color= ['C0','C1','C2','C3','C4'])
plt.xticks(range(s+1), scripts, rotation='vertical')
plt.show()

