# -*- coding: utf-8 -*-
"""
Created on Wed May 16 21:30:25 2018

@author: yanghe
"""

import tensorflow as tf
from sklearn.preprocessing import scale
import numpy as np
import pandas as pd

training_data = pd.read_csv('trainingData.csv', header=0)
test_data = pd.read_csv('validationData.csv', header=0)

#print(training_data.head())
#print(test_data.head())



train_x = scale(np.asarray(training_data.ix[:, 0:520]))
train_y = np.asarray(training_data["BUILDINGID"].map(str) + training_data["FLOOR"].map(str))  
train_y = np.asarray(pd.get_dummies(train_y))  

test_x = scale(np.asarray(test_data.ix[:,0:520]))  
test_y = np.asarray(test_data["BUILDINGID"].map(str) + test_data["FLOOR"].map(str))  
test_y = np.asarray(pd.get_dummies(test_y))  

print('train_x.shape',train_x.shape)
print('train_y.shape',train_y.shape)
print('test_x.shape',test_x.shape)
print('test_x.shape',test_y.shape)

def get_weight(shape, lambda1):
    weight = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda1)(weight))
    bias = tf.Variable(tf.constant(0.1, shape=[shape[1]]))
    return weight, bias


def encoder(x, layer_dimension):  
    cur_layer = x
    n_layers = len(layer_dimension)
    in_dimension = layer_dimension[0]
    for i in range(1, n_layers):
        out_dimension = layer_dimension[i]
        weight, bias = get_weight([in_dimension, out_dimension], 0.3)
        cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
        in_dimension = layer_dimension[i]
    return cur_layer

def decoder(x, layer_dimension):
    decoder_layer = encoder(x, layer_dimension)
    return decoder_layer

def train(epoch,layer_dimension):
    print('train....')
    x = tf.placeholder(tf.float32, shape=(None, input_layer))
    y = tf.placeholder(tf.float32, shape=(None, out_layer))
    
    encoder_x = encoder(x, layer_dimension)
    dencoder_x =  decoder(encoder_x, layer_dimension[::-1])
    weight, bias = get_weight([layer_dimension[-1], 13], 0.3)
    y_ = tf.nn.softmax(tf.matmul(encoder_x, weight) + bias)
    cost_mes = tf.reduce_mean(tf.pow(x - dencoder_x, 2))
    cross_entropy = -tf.reduce_sum(y * tf.log1p(y_))
    cost_mes_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost_mes)  
    cross_entropy_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cross_entropy)  
    
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #train_op = tf.train.AdamOptimizer(0.1).minimize(loss)
    #with tf.Graph().as_default() as g:
    #tf.reset_default_graph()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for epoch in range(epoch):
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = train_x[start:end]
                batch_y = train_y[start:end]
                _ = sess.run([cost_mes_optimizer], feed_dict={x:batch_x})
                i += batch_size
                if (epoch % 2 == 0 and i % (batch_size * 100) == 0) :
                    print('准确率: ', accuracy.eval({x:test_x, y:test_y})) 
        
        for epoch in range(epoch):
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = train_x[start:end]
                batch_y = train_y[start:end]
                _ = sess.run([cross_entropy_optimizer], feed_dict={x:batch_x, y:batch_y})
                i += batch_size
                if (epoch % 2 == 0 and i % (batch_size * 100) == 0) :
                    print('准确率: ', accuracy.eval({x:test_x, y:test_y})) 
                
input_layer = 520
out_layer = 13
layer_dimension = [input_layer,256, 100]
batch_size = 10
epoch = 10


x = tf.placeholder(tf.float32, shape=(None, input_layer))
y = tf.placeholder(tf.float32, shape=(None, out_layer))

train(epoch,layer_dimension)