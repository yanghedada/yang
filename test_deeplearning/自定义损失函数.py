# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 08:09:33 2018

@author: yanghe
"""

import tensorflow as tf
import numpy as np
batch_size = 8
rdm = np.random.RandomState(1)
dataset_size = 128
steps = 5000

x = tf.placeholder(tf.float32 ,shape=[None ,2], name='x-input')
y_ = tf.placeholder(tf.float32 ,shape=[None ,2], name='y-input')


w1 = tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
y = tf.matmul(x ,w1)

#==============================================================================
#        { a (x - y) ,  x >y
# loss = 
#        { b( y -x)  , x < y
#==============================================================================
loss_less  = 10
loss_more = 1

loss = tf.reduce_mean(tf.where(tf.greater(y , y_) ,
                                (y - y_) *loss_more ,
                                (y_ - y)*loss_less))

train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
X  = rdm.rand(dataset_size , 2)

Y = [[x1  , x2 + np.random.rand() / 10 - 0.05 ] for (x1 , x2 ) in X ]

      
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(steps):
        start =   (i * batch_size ) % dataset_size
        end = min(start + batch_size , dataset_size)
        loss_ , _ = sess.run([loss , train_op] ,
                             feed_dict={x:X[start:end] , y_:Y[start : end]})
        if i % 100 == 0 :
            print('loss is :',loss_)
    print('w1 : ',sess.run(w1))
    

