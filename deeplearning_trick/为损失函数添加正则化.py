# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 09:08:34 2018

@author: yanghe
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    data = []
    label = []
    np.random.seed(0)
    for i in range(150):
        x1 = np.random.uniform(-1,1)
        x2 = np.random.uniform(0,2)
        if x1**2 + x2**2 <= 1:
            data.append([np.random.normal(x1, 0.1),np.random.normal(x2,0.1)])
            label.append(0)
        else:
            data.append([np.random.normal(x1, 0.1), np.random.normal(x2, 0.1)])
            label.append(1)
    return np.reshape(data , (-1,2)) , np.reshape(label,(-1,1))
    
data , label = load_data()

#==============================================================================
# plt.scatter(data[:,0], data[:,1], c=label,
#            cmap="RdBu", vmin=-.2, vmax=1.2, edgecolor="white")
# plt.show()
#==============================================================================


def get_weight_l2(shape , l2=False):
    weight = tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    if l2 == True :
        tf.add_to_collection('loss' ,tf.contrib.layers.l2_regularizer(0.01)(weight))
    bias = tf.Variable(tf.random_normal([shape[1]]))
    return bias , weight
def inference( x , shape):
    n_layer = x
    n_shape = len(shape)
    in_dimension = shape[0]
    for i in range(1, n_shape):
        out_dimension = shape[i]
        bias , weight = get_weight_l2([in_dimension , out_dimension])
        
        n_layer = tf.nn.relu(tf.matmul(n_layer , weight) + bias)
        in_dimension = shape[i]
    return n_layer
    
x = tf.placeholder(tf.float32 , [None , 2])
y_ = tf.placeholder(tf.float32 , [None , 1])
n_depth = [2 , 8 , 1]

y  = inference(x ,n_depth)

mse_loss = tf.reduce_mean(tf.square(y_ - y))

tf.add_to_collection('loss' , mse_loss)

to_loss = tf.add_n(tf.get_collection('loss'))

train_op = tf.train.AdamOptimizer(0.01).minimize(to_loss)

init = tf.global_variables_initializer()

batch_size = 30
dataset_size = len(data)

with tf.Session()  as sess:
    
    sess.run(init)
    for i in range(100):
        start =   (i * batch_size ) % dataset_size
        end = min(start + batch_size , dataset_size)
        loss_ , _ = sess.run([to_loss,train_op] ,
                             feed_dict={x:data[start:end] , y_:label[start : end]})
  
        print('after %d steps , loss: %f'%(i , loss_))
    xx, yy = np.mgrid[-1:1:.01, 0:2:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = sess.run(y, feed_dict={x:grid})
    probs = probs.reshape(xx.shape)
    
    plt.scatter(data[:,0], data[:,1], c=label,
               cmap="RdBu", vmin=-.2, vmax=1.2, edgecolor="white")
    plt.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.1)
    plt.show()
        


