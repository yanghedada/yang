# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 08:42:02 2018

@author: yanghe
"""
import tensorflow as tf
import numpy as np

#最小化函数  y=x2y=x2 , 选择初始点  x0=5
training_steps = 300
x = tf.Variable(tf.constant(5.0),name='x-input')
y = tf.square(x)
global_steps = tf.Variable(0)
learining_rate = tf.train.exponential_decay(0.1 , global_steps , 1 ,0.96,staircase=True)
#trian_op = tf.train.GradientDescentOptimizer(0.1).minimize(y)
trian_op = tf.train.AdamOptimizer(learining_rate).minimize(y)
#trian_op = tf.train.AdadeltaOptimizer(0.1).minimize(y)


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(training_steps):
        sess.run(trian_op)
        print('iteration %s :y is %s'%(i , sess.run(x)))
    


