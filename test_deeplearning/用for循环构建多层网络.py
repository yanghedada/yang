# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 20:25:28 2018

@author: yanghe
"""

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(r'E:\python\mnist_data', one_hot=True)

def get_weight(shape):
    weight = tf.Variable(tf.random_normal(shape))
    bias = tf.Variable(tf.random_normal([shape[1]]))
    return bias , weight
    
def inference( x , shape):
    n_layer = x
    n_shape = len(shape)
    in_dimension = shape[0]
    for i in range(1, n_shape):
        out_dimension = shape[i]
        bias , weight = get_weight([in_dimension , out_dimension])
        
        n_layer = tf.nn.relu(tf.matmul(n_layer , weight) + bias)
        in_dimension = shape[i]
    return n_layer
    
x = tf.placeholder(tf.float32 , [None , 784])
y_ = tf.placeholder(tf.float32 , [None , 10])
n_depth = [784 , 128 , 10]

y  = inference(x ,n_depth)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
cross_entropy_mean = tf.reduce_mean(tf.cast(cross_entropy,tf.float32))

train_op = tf.train.AdamOptimizer(0.01).minimize(cross_entropy_mean)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


init = tf.global_variables_initializer()
training_iters = 100
batch_size = 4

with tf.Session()  as sess:
    
    sess.run(init)
    
    bx , by = mnist.train.next_batch(batch_size)
    max_op = int(mnist.train.num_examples / batch_size)
    for i in range(training_iters):
        for j in range(max_op):
            loss , _ = sess.run([cross_entropy_mean , train_op] , 
                                feed_dict={x : bx , y_:by})
        if i % 2 == 0:
            print(loss)
    print('accuracy : ',sess.run(accuracy ,feed_dict={x:mnist.test.images[:100],y_:mnist.test.labels[:100]}))
    
    