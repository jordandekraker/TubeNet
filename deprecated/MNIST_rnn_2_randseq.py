""" Recurrent Neural Network.
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

import fixeye_saccade as fe
import tensorflow as tf
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Training Parameters
learning_rate = 0.001
training_steps = 10000
batch_size = 1
display_step = 2000

# Network Parameters
num_input = 200 # MNIST data input (img shape: 28*28); new shape 10*10 + 100 units fixation
timesteps = 10 # timesteps
#fixseq = np.array([55, 75, 95, 35, 15, 55, 57, 59, 53, 51])
nfixes = 10

num_hidden = 400 # hidden layer num of features
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Define weights
topweights = {'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))}
topbiases = {'out': tf.Variable(tf.random_normal([num_classes]))}
#other variables to initalize
batch_x = np.empty([batch_size, timesteps, num_input])
     
# Define the network                    
lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0, reuse=None)
outputs, states = tf.nn.dynamic_rnn(lstm_cell, X, dtype=tf.float32)
#can make reusable only after dynamic_rnn call
lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0, reuse=True) 
outseq = tf.matmul(outputs[:,-1,:], topweights['out']) + topbiases['out']
logits = outseq
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
merged = tf.summary.merge_all()

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
# Start training
sess = tf.Session() 
# Run the initializer
train_writer = tf.summary.FileWriter('tmp/MNIST_rnn_2_randseq',sess.graph)
sess.run(init)
#saver.restore(sess, "/tmp/MNIST_rnn_2_randseq.ckpt")

newfixind = (np.random.rand(training_steps*batch_size*nfixes)*100).astype(int)
o=0
for step in range(training_steps):
    batch_x_old, batch_y = mnist.train.next_batch(batch_size)
    # Apply fisheye filter and reshape data
    for n in range(batch_size):
        for f in range(nfixes):
            img = batch_x_old[n,:]
            img = img.reshape(28,28)
            fix = np.zeros([1,100])
            fix[0,newfixind[o]] = 1;
            batch_x[n,f,range(num_input)] = np.concatenate((fe.fixeye(img,fix),fix),1)
            o=o+1
    # Run optimization op (backprop)
    sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
    if step % display_step == 0 or step == 0:
        # Calculate batch loss and accuracy
        loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                             Y: batch_y})
#        train_writer.add_summary(summ, step)
        print("Step " + str(step) + ", Minibatch Loss= " + \
              "{:.4f}".format(loss) + ", Training Accuracy= " + \
              "{:.3f}".format(acc))

print("Optimization Finished!")
saver = tf.train.Saver()
saver.save(sess, "tmp/MNIST_rnn_2_randseq.ckpt")

# Calculate accuracy for 128 mnist test images
test_len = 128
test_data_old = mnist.test.images[:test_len]
test_data = np.empty([test_len, timesteps, num_input])
for n in range(test_len):
    for f in range(nfixes):
        img = test_data_old[n,:]
        img = img.reshape(28,28)
        fix = np.zeros([1,100])
        newfixind = (np.random.rand(1)*100).astype(int)
        fix[0,newfixind] = 1;
        test_data[n,f,range(num_input)] = np.concatenate((fe.fixeye(img,fix),fix),1)

test_label = mnist.test.labels[:test_len]
print("Testing Accuracy:", \
    sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))