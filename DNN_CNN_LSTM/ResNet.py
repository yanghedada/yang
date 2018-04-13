# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 18:33:43 2018

@author: yanghe
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 09:25:26 2018

@author: yanghe
"""

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data



def identity_block(X,filters,stage,block,keep_prob):
    conv_name_base = 'res' + str(stage) + block + '_brach'
    
    F1 , F2 ,F3 = filters    
    
    weight = {
              conv_name_base+'w1':tf.Variable(tf.truncated_normal(shape=[3,3,F1,F2],stddev=0.1),name=conv_name_base+'w1'),
              conv_name_base+'w2':tf.Variable(tf.truncated_normal(shape=[3,3,F2,F3],stddev=0.1),name=conv_name_base+'w2'),
            }
    bisases={conv_name_base+'ec1':tf.Variable(tf.random_normal([F2]),name=conv_name_base+'ec1'),
             conv_name_base+'ec2':tf.Variable(tf.random_normal([F3]),name=conv_name_base+'ec2'),
             }
    conv1 = tf.nn.relu(tf.nn.conv2d(X , weight[conv_name_base+'w1'],strides=[1,1,1,1],padding='SAME') + bisases[conv_name_base+'ec1'])
    conv1_drop = tf.nn.dropout(conv1 , keep_prob)
    
    conv2 = tf.nn.conv2d(conv1_drop, weight[conv_name_base+'w2'],strides=[1,1,1,1],padding='SAME') + bisases[conv_name_base+'ec2']
    conv2_drop = tf.nn.dropout(conv2, keep_prob)
    
    out = tf.nn.relu(conv2_drop + X)
    
    return out

def convolution_block(X,filters,stage,block,keep_prob):
    conv_name_base = 'res' + str(stage) + block + '_brach'
    
    F1 , F2  = filters    
    

    weight = {
              conv_name_base+'w1':tf.Variable(tf.truncated_normal(shape=[3,3,F1,F2],stddev=0.1),name=conv_name_base+'w1'),
              }
    bisases={
             conv_name_base+'ec1':tf.Variable(tf.random_normal([F2]),name=conv_name_base+'ec1'),
             
             }
    conv1 = tf.nn.relu(tf.nn.conv2d(X , weight[conv_name_base+'w1'],strides=[1,1,1,1],padding='SAME') + bisases[conv_name_base+'ec1'])
    conv1_pool = tf.nn.max_pool(conv1 , ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv1_drop = tf.nn.dropout(conv1_pool , keep_prob)
    
    return conv1_drop 
    
def ResNet(X,keep_prob):
    X = tf.reshape(X,(-1,28,28,1))
    
    weight_  = {
              'w0':tf.Variable(tf.truncated_normal(shape=[3,3,1,32],stddev=0.1),name='w0'),
              'w1':tf.Variable(tf.truncated_normal(shape=[512,1024],stddev=0.1),name='w1'),
              'w2':tf.Variable(tf.truncated_normal(shape=[1024,10],stddev=0.1),name='w2'),
              }
    bisases_ ={
             'ec1':tf.Variable(tf.random_normal([1024]),name='ec1'),
             'ec2':tf.Variable(tf.random_normal([10]),name='ec2'),
             
             }
    X = tf.nn.relu(tf.nn.conv2d(X , weight_['w0'],strides=[1,1,1,1],padding='SAME'))
    X = identity_block(X ,filters=[32,32,32] ,stage=1 ,block='a',keep_prob=0.8 )
    X = convolution_block(X ,filters=[32,64] ,stage=1 ,block='b',keep_prob=0.8 )
  
    X = identity_block(X ,filters=[64,64,64] ,stage=2 ,block='a',keep_prob=0.8 )
    X = convolution_block(X ,filters=[64,32] ,stage=2 ,block='b',keep_prob=0.8 )
    
    
    X = identity_block(X ,filters=[32,32,32] ,stage=3 ,block='a',keep_prob=0.8 )
    X = convolution_block(X ,filters=[32,32] ,stage=3 ,block='b',keep_prob=0.8 )
   
    X = tf.reshape(X ,(-1,512))
    
    
             
    X = tf.nn.relu(tf.matmul(X , weight_['w1']) + bisases_['ec1'])
    X = tf.nn.relu(tf.matmul(X ,weight_['w2'] + bisases_['ec2']))
    
    return X
    
    
X = tf.placeholder(tf.float32,[None ,784])
y = tf.placeholder(tf.float32,[None ,10])
keep_prob= tf.placeholder(tf.float32)

y_pred = ResNet(X, keep_prob)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)


correct_pred = tf.equal(tf.argmax(y_pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


mnist = input_data.read_data_sets(r'E:\python\mnist_data', one_hot=True)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
#==============================================================================
#     total_batch = int(mnist.train.num_examples/128)
#     # Training cycle
#     for epoch in range(6):
#         # Loop over all batches
#         for i in range(total_batch):
#             batch_xs, batch_ys = mnist.train.next_batch(128)
#             # Run optimization op (backprop) and cost op (to get loss value)
#             _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs,y:batch_ys,keep_prob:0.8})
#         # Display logs per epoch step
#         if epoch % 2 == 0:
#             acc = sess.run(accuracy, feed_dict={X: batch_xs, y: batch_ys, keep_prob: 1.})
# 
#             print("Epoch:", '%04d' % (epoch+1),
#                   "cost=", "{:.9f}".format(c),
#                     "Training Accuracy= " + "{:.5f}".format(acc))
#     print ("Optimization Finished!")
#     # Calculate accuracy for 256 mnist test images
#==============================================================================
    print ("Testing Accuracy:", sess.run(accuracy, feed_dict={X: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.}))

    
    
    
    