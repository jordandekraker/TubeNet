#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 14:29:24 2018

Creates TubeNet architecture for classifying MNIST images. 
The goal of this architecture is to have sensory, memory, and motor processing 
be fully integrated at the most basic level possible, similar to a primitive 
organism (inspired by the ontogeny/phylogeny structure of the 'neural tube'). 
Added complexity can then be added via additional units, layers, or additional
units or layers nested in the LSTM hidden layer.

@author: jordandekraker
"""

try:
    loaded
except NameError:
    maxfixes = 50
    minfixes = 5
    iters = 1000
    disp_n_iters = int(iters/100)
    test_len = 100
    
    inputshape = 25
    fixshape = 100
    memshape = 200
    outshape = 10
    
    
    import numpy as np
    import tensorflow as tf
    import fixeye_saccade as fe
    import matplotlib.pyplot as plt
    # Import MNIST data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    loaded = True
else:
    print('inputs already defined')

# tf Graph setup
X = tf.placeholder("float32", [1,784]) #size of an MNIST image
Y = tf.placeholder("float64", [1,10]) #number of possible classes
e = tf.placeholder("float64", [1])

# initialize tensorflow trainable variables
# S=sensory, H=hippocampal, M=motor
# w=weights, b=biases, a=activations
S1w = tf.Variable(tf.random_normal([inputshape+fixshape, 2*(inputshape+fixshape)])) 
S1b = tf.Variable(tf.random_normal([2*(inputshape+fixshape)])) 
with tf.variable_scope('rnn', reuse=tf.AUTO_REUSE):
    with tf.variable_scope('basic_lstm_cell'):
        weights = tf.get_variable('kernel',[(2*(inputshape+fixshape)+memshape), (memshape)*4])
        biases = tf.get_variable('bias',[(memshape)*4])         
H1 = tf.contrib.rnn.BasicLSTMCell(memshape, state_is_tuple=True, reuse=True)
H2w = tf.Variable(tf.random_normal([memshape,outshape])) 
H2b = tf.Variable(tf.random_normal([outshape])) 
M1w = tf.Variable(tf.random_normal([outshape+fixshape, fixshape])) 
M1b = tf.Variable(tf.random_normal([fixshape])) 

H1m_old_m = tf.Variable(tf.random_normal([1, memshape])) # ititial hidden layer c_state and m_state 
H1m_old_c = tf.Variable(tf.random_normal([1, memshape])) # ititial hidden layer c_state and m_state 
RollingAverage = tf.Variable(tf.zeros([maxfixes]))
fix_list = tf.Variable(tf.sparse_tensor_to_dense(tf.SparseTensor([[55]],[1.0],[fixshape])),trainable=False) # starting point

optimizer_FIX = tf.train.GradientDescentOptimizer(learning_rate=0.1) 
optimizer_MNIST = tf.train.GradientDescentOptimizer(learning_rate=0.001)



# define the model
def tubenet(X,Y,e):
    H2a = tf.constant([0.0],shape=[1,10])
    certainty = tf.constant([0.0],shape=[maxfixes])
    n = tf.constant([0])
        
    # get new series of fixations by feeding forward for n fixations (in tf.while_loop)
    def explore_cond(n,certainty,H2a):
        # check whether certainty is still increasing (over 5 trial window)
        def increase_decrease(): return tf.reduce_sum(tf.subtract(certainty[n[0]-4:n[0]],certainty[n[0]-5:n[0]-1]))
        def canttellyet(): return tf.constant(0.0)
        certaintycheck1 = tf.cond(n[0]>minfixes+1, increase_decrease,canttellyet)
        certaintycheck2 = tf.cond( (certaintycheck1>-0.01),lambda:True,lambda:False)
        return tf.cond( (n[0]<minfixes) | (certaintycheck2) & (n[0]<maxfixes),lambda:True,lambda:False)
    
    def explore(n,certainty,H2a):
        #get fisheye image at fix, and concatenate fix
        FEimg = tf.cast(tf.py_func(fe.fixeye,[X, fix_list],[tf.float64]),tf.float32)
        fixwithimg = tf.reshape(tf.concat((FEimg[0,:],fix_list),0),[1,inputshape+fixshape])
        
        # feed new image and its fix forward
        S1a = tf.tanh(tf.matmul(fixwithimg, S1w) + S1b) 
        with tf.variable_scope('rnn') as scope:
            scope.reuse_variables()
            H1a, H1m = H1(S1a, (H1m_old_m,H1m_old_c))
        H2a = tf.tanh(tf.matmul(H1a, H2w) + H2b)
        # concatenate fix after LSTM layer
        H2aWithfix = tf.reshape(tf.concat((H2a[0,:],fix_list),0),[1,-1])
        M1a = tf.matmul(H2aWithfix, M1w) + M1b # linear activation function?
        
        # measure certainty to determine whether to keep exploring
        shape = certainty.shape
        certainty = tf.concat([certainty[:n[0]+1],tf.reduce_max(tf.nn.softmax(H2a),1),certainty[n[0]+2:]],0)
        certainty.set_shape(shape)

        # train for fixes that increase the change in memory 
        Qsignal = tf.reduce_mean(tf.square(H1m[1]-H1m_old_c),1) # mean mem diff
        Q = tf.reshape(M1a[0,:],[fixshape])
        Qchange = Qsignal-RollingAverage[n[0]] #difference from rollingaverage
        Qtarget = Q + tf.multiply(Q,Qchange)
        loss_FIX = tf.squared_difference(Q,Qtarget)
        
        # sometimes try random fixation (e decreases over time)
        fixind = tf.cond(tf.random_uniform([1],0,1,dtype=tf.float64)[0] < e[0], 
                         lambda: tf.random_uniform([1],0,fixshape,dtype=tf.int64),
                         lambda: [tf.argmax(Q)])
        
        #regroup for next fixation
        op1 = optimizer_FIX.minimize(loss_FIX,var_list=[M1w,M1b])
        op2 = RollingAverage[n[0]+1].assign(RollingAverage[n[0]]*0.99 + Qsignal[0]*0.01)
        op3 = tf.assign(fix_list,tf.sparse_tensor_to_dense(tf.SparseTensor([fixind],[1.0],[fixshape])))
        op4 = H1m_old_m.assign(H1m[0])
        op5 = H1m_old_c.assign(H1m[1])
        
        with tf.control_dependencies([op1,op2,op3,op4,op5]):
            n = n+1
        return n,certainty,H2a
    
    n,certainty,H2a = tf.while_loop(explore_cond,explore,[n,certainty,H2a])
#        shape_invariants=[tf.TensorShape([None]),tf.TensorShape([1]),tf.TensorShape([None]),tf.TensorShape([1,10])
    
    # Define loss and optimizer for MNIST
    logits = tf.reshape(H2a[0,:outshape],[1,10]) # MNIST categories
    loss_MNIST = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits, labels=Y))
    op6 = optimizer_MNIST.minimize(loss_MNIST)
    with tf.control_dependencies([op6]):
        # Evaluate model with MNIST (with test logits, for dropout to be disabled)
        prediction = tf.nn.softmax(logits)
        correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(Y,1))
        accuracy_MNIST = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    return loss_MNIST,accuracy_MNIST,n
loss_MNIST,accuracy_MNIST,n = tubenet(X,Y,e)



# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()
#saver.restore(sess, "tmp/TubeNet_MNIST_nfixes_singleoptimizer.ckpt")
#writer = tf.train.write_graph(tf.get_default_graph(),'tmp/tensorboard','dualoptimizer')

# Start training
# initialize outer np variables
lossperiter = np.zeros(iters)
accuracyperiter = np.zeros(iters)
nn = np.zeros(iters)
for i in range(iters): # iterations
    ee = 1./((i/50) + 10)
    MNISTimg, MNISTcat = mnist.train.next_batch(1)
    lossperiter[i],accuracyperiter[i],nn[i] = sess.run(
            [loss_MNIST,accuracy_MNIST,n], 
            feed_dict={X: MNISTimg, Y: MNISTcat, e: [ee]})
    
    # ongoing display
    if i % disp_n_iters == 0:
        # Calculate seq loss and accuracy and see fixations used
        print("Iteration " + str(i) + ", loss_MNIST= " + \
              "{:.4f}".format(np.mean(lossperiter[i-disp_n_iters:i])) + ", Accuracy= " + \
              "{:.4f}".format(np.mean(accuracyperiter[i-disp_n_iters:i])) + 
              ", n fixes was:" + str(nn[i]))
#        print(np.where(finalfix[:,:]==1)[1])
    
print("Optimization Finished!")
#saver.save(sess, "tmp/TubeNet_MNIST_nfixes_singleoptimizer.ckpt")
#train_writer = tf.summary.FileWriter('tmp/TubeNet_MNIST_nfixes_singleoptimizer',sess.graph)

meanaccperiter = np.zeros(int(iters/disp_n_iters))
mean_n = np.zeros(int(iters/disp_n_iters))
for n in range(int(iters/disp_n_iters)):
    meanaccperiter[n] = np.mean(accuracyperiter[n*disp_n_iters : (n+1)*disp_n_iters])
    mean_n[n] = np.mean(nn[n*disp_n_iters : (n+1)*disp_n_iters])
plt.plot(meanaccperiter)
plt.show()
plt.plot(mean_n)
plt.show()


# Calculate accuracy for 128 mnist test images
finallossperiter = np.zeros(test_len)
finalaccuracyperiter = np.zeros(test_len)
for i in range(test_len):
    MNISTimg, MNISTcat = mnist.test.next_batch(1)
    finallossperiter[i], finalaccuracyperiter[i] = sess.run(
            [loss_MNIST,accuracy_MNIST],
            feed_dict={X: MNISTimg, Y: MNISTcat, e: [ee]})
             
print("Test mean loss_MNIST= " + "{:.4f}".format(np.mean(finallossperiter)) \
      + ", Accuracy= " + "{:.4f}".format(np.mean(finalaccuracyperiter)))
#print(np.where(outerfixes[:,:]==1)[1])

sess.close()