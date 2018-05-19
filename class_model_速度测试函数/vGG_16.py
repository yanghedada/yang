# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 18:03:52 2018

@author: yanghe
"""

import tensorflow as tf
import math
import time
from datetime import datetime

def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def conv_op(input_op ,name ,kh ,kw ,n_out ,dh ,dw ,p):
    n_in = input_op.get_shape().as_list()[-1]
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+'w',
                                 shape=[kh, kw, n_in, n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, (1,dh,dw,1),padding='SAME')
        biases = tf.get_variable(scope+'b',
                                 [n_out],
                                 initializer=tf.constant_initializer(0.01))
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        print_activations(activation)
        return activation

        
        
def fc_op(input_op ,name ,n_out, p):
    n_in = input_op.get_shape().as_list()[-1]
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+'w',
                                 shape=[n_in,n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable(scope+'b',
                                 shape=[n_out],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.01))
        z = tf.matmul(input_op, kernel) + biases
        activation = tf.nn.bias_add(z, biases)
        p += [kernel, biases]
        print_activations(activation)
        return activation

def mpool_op(input_op, name, kh, kw, dh, dw):
    max_pool = tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='SAME',  
                          name=name)
    print_activations(max_pool)
    return max_pool
    
def inference_op(input_op, keep_prob):
    p = []
    conv1_1 = conv_op(input_op, name='conv1_1', kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    conv1_2 = conv_op(conv1_1, name='conv1_2', kh=3, kw=3 ,n_out=64, dh=1, dw=1, p=p)
    pool1 = mpool_op(conv1_2, name='pool1', kh=2, kw=2, dw=2, dh=2)
    
    conv2_1 = conv_op(pool1, name='conv2_1', kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    conv2_2 = conv_op(conv2_1, name='conv2_2', kh=3, kw=3 ,n_out=128, dh=1, dw=1, p=p)
    pool2 = mpool_op(conv2_2, name='pool2', kh=2, kw=2, dw=2, dh=2)
    
    conv3_1 = conv_op(pool2, name='conv3_1', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_2 = conv_op(conv3_1, name='conv3_2', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_3 = conv_op(conv3_2, name='conv3_3', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    pool3 = mpool_op(conv3_3, name='pool3', kh=2, kw=2, dw=2, dh=2)
    
    conv4_1 = conv_op(pool3, name='conv4_1', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_2 = conv_op(conv4_1, name='conv4_2', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_3 = conv_op(conv4_2, name='conv4_3', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool4 = mpool_op(conv4_3, name='pool4', kh=2, kw=2, dw=2, dh=2)
    
    conv5_1 = conv_op(pool4, name='conv5_1', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_2 = conv_op(conv5_1, name='conv5_2', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_3 = conv_op(conv5_2, name='conv5_3', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool5 = mpool_op(conv5_3, name='pool5', kh=2, kw=2, dw=2, dh=2)
    
    shp = pool5.get_shape().as_list()
    flattend_shape = shp[1] *  shp[2] * shp[3]
    resh1 = tf.reshape(pool5, [-1, flattend_shape], name='resh1')
    
    fc6 = fc_op(resh1, name='fc6',n_out=4096, p=p)
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name='fc6_drop')
    
    fc7 = fc_op(fc6_drop, name='fc7', n_out=4096, p=p)
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name='fc7_drop')
    
    fc8 = fc_op(fc7_drop, name='fc8', n_out=1000, p=p)
    softmax = tf.nn.softmax(fc8)
    
    prediction = tf.argmax(softmax, 1)
    return prediction, softmax ,fc8, p 
    
def time_tensroflow_run(session, target, feed, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target, feed_dict=feed)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s : ste%d , duration =%.3f'%
                      (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print ('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
         (datetime.now(), info_string, num_batches, mn, sd))
def run_benbhmark():
    with tf.Graph().as_default():
        image_size = 28
        image = tf.Variable(tf.random_normal([batch_size,
                                              image_size,
                                              image_size,3],
                                              dtype=tf.float32,
                                              stddev=1e-1))    
    
        keep_prob = tf.placeholder(tf.float32)
        
        prediction, softmax ,fc8, p = inference_op(image, keep_prob)
        
        init = tf.global_variables_initializer()
        sess= tf.Session()
        sess.run(init)
        time_tensroflow_run(sess, prediction, {keep_prob:1.}, 'forward')
        
        objective = tf.nn.l2_loss(fc8)
        grad = tf.gradients(objective, p)
        
        time_tensroflow_run(sess, grad, {keep_prob:1.}, 'forward-backward')
        
batch_size= 32
num_batches = 10
    
run_benbhmark()


