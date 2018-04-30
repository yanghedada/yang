# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 20:14:38 2018

@author: yanghe
"""

import numpy as np
import tensorflow as tf
import pygame  
from  voice_tool import *

pygame.mixer.init() 
track4=pygame.mixer.Sound("./raw_data/dev/2.wav")
track4.play()


X = np.load("./XY_train/X.npy")
Y = np.load("./XY_train/Y.npy")
Y = np.reshape(Y, (-1, 1375))
X_dev = np.load("./XY_dev/X_dev.npy")
Y_dev = np.load("./XY_dev/Y_dev.npy")
Y_dev = np.reshape(Y_dev, (-1, 1375))
def get_weight_l2(shape ,name, regulariztion_rate=None):
    weight = tf.get_variable('weight_%s'%str(name), shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1) )
    bias =  tf.get_variable('bias_%s'%str(name), shape=[shape[-1]], initializer=tf.constant_initializer(0.1) )
    if regulariztion_rate != None  :
        tf.add_to_collection('loss' ,tf.contrib.layers.l2_regularizer(regulariztion_rate)(weight))
    return bias , weight

def model(inputs):
    inputs = tf.reshape(inputs, (-1, 5511, 1, 101))
    with tf.variable_scope('conv1'):
        conv1_biases,conv1_weight = get_weight_l2(shape=[15, 1, 101, 128],name='layer1')
        conv1 = tf.nn.conv2d(inputs, conv1_weight, strides=[1,1,1,1], padding='SAME') 
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1 , conv1_biases))
        pool1 = tf.nn.max_pool(relu1, ksize=[1,15,1,1], strides=[1,4,1,1], padding='VALID')
        dorp1 = tf.nn.dropout(pool1,keep_prob )
        dorp1 = tf.reshape(dorp1, (-1, 1375, 128))
        dorp1 = tf.transpose(dorp1, [1, 0, 2])
    with tf.variable_scope('rnn'):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(128, forget_bias=1.0)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * 2)
        outputs, _ = tf.nn.dynamic_rnn(cell, dorp1, dtype=tf.float32)
        outputs = tf.reshape(outputs, (-1, 1375, 128, 1))
    with tf.variable_scope('conv2'):
        conv2_biases,conv2_weight = get_weight_l2(shape=[1, 128, 1, 1],name='layer2')
        conv2 = tf.nn.conv2d(outputs, conv2_weight, strides=[1,1,128,1], padding='SAME') 
        sig1 = tf.nn.sigmoid(tf.nn.bias_add(conv2 , conv2_biases))
        print(sig1.shape)
        sig1 = tf.reshape(sig1, (-1, 1375))
        print(sig1.shape)
    return sig1

def train():
    global_step = tf.Variable(0, trainable=False)
    with tf.variable_scope("moving_average"):
        variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.variable_scope('voice') as scope:
        pred = model(input_data)
    with tf.variable_scope("train_step"):
        learning_rate = tf.train.exponential_decay(
                        learning_rate_base,
                        global_step,
                        1,
                        learing_rate_decay,
                        staircase=True)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum( targets* tf.log(pred) + (1-targets)* tf.log(1-pred)  ,reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
    
    with tf.control_dependencies([train_step , variables_averages_op]):
        train_op = tf.no_op(name='train')
    
    saver=tf.train.Saver()
    with tf.Session() as sess :
        tf.global_variables_initializer().run()
        for i in range(training_steps):
            _= sess.run(train_op , feed_dict={input_data:X , targets:Y,keep_prob:0.8})
            if i % 2 == 0 :
                loss = sess.run(cross_entropy , feed_dict={input_data:X_dev , targets:Y_dev,keep_prob:1.0})
                print("After %d training step(s), the model loss is %g " % (i, loss))
        saver.save(sess , 'saver/moedl_voce_3.ckpt')
            
def predict():
    filename  = "./raw_data/dev/2.wav"
    x = graph_spectrogram(filename)
    x  = x.swapaxes(0,1)
    x = np.expand_dims(x, axis=0)
    with tf.variable_scope('voice') as scope:
        scope.reuse_variables()
        prediction = model(input_data)
    saver=tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver.restore(sess,'saver/moedl_voce_1.ckpt')
        predict_= sess.run(prediction,feed_dict={input_data:x,keep_prob:1.0})
        print(predict_)
        chime_on_activate(filename, predict_, 0.5)
        track4=pygame.mixer.Sound("./chime_output.wav")
        track4.play()
        
keep_prob = tf.placeholder(tf.float32)
input_data = tf.placeholder(tf.float32, [None,  5511, 101])
targets = tf.placeholder(tf.float32,[None, 1375])
training_steps = 50 
learning_rate_base = 0.01
learing_rate_decay = 0.99
moving_average_decay = 0.99

#train()
#predict()


#==============================================================================
# filename  = "./raw_data/dev/2.wav"
# x = graph_spectrogram(filename)
# x  = x.swapaxes(0,1)
# x = np.expand_dims(x, axis=0)
# with tf.variable_scope('voice') as scope:
#     scope.reuse_variables()
#     prediction = model(input_data)
# saver=tf.train.Saver()
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     saver.restore(sess , 'saver/moedl_voce_3.ckpt')
#     predict_= sess.run(prediction,feed_dict={input_data:x,keep_prob:1.0})
#     chime_on_activate(filename, predict_,  0.9993212)
#     track4=pygame.mixer.Sound("./chime_output.wav")
#     track4.play()
#==============================================================================

