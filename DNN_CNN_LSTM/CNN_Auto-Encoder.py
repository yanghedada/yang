# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 12:39:59 2018

@author: yanghe
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

train_iter = 2
learning_rate = 0.01
noise_factor = 0.5
batch_size  = 128

X = tf.placeholder(tf.float32,[None ,784])
keep_prob = tf.placeholder(tf.float32)



weight = {'enwc1':tf.Variable(tf.truncated_normal(shape=[3,3,1,64],stddev=0.1)),
          'dewc1':tf.Variable(tf.truncated_normal(shape=[3,3,1,64],stddev=0.1)),
          }
bisases={'ecb1':tf.Variable(tf.random_normal([64])),
         'deb1':tf.Variable(tf.random_normal([1])),
        }


def enconv(X , w ,b,keep_prob,batch_size):
    X  = tf.reshape(X , shape=[batch_size, 28 ,28 ,1])
    conv1 = tf.nn.relu(tf.nn.conv2d(X , w['enwc1'],strides=[1,1,1,1],padding='SAME') + b['ecb1'])
    conv1_pool = tf.nn.max_pool(conv1 , ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv1_drop = tf.nn.dropout(conv1_pool , keep_prob)
    return conv1_drop
    
def deconv(conv1_drop , w ,b,keep_prob,batch_size): 
    
    decon1 = tf.nn.relu(tf.nn.conv2d_transpose(conv1_drop, w['dewc1'],output_shape=[batch_size,28,28,1],strides=[1,2,2,1],padding='SAME') + b['deb1'])
    decon1_drop = tf.nn.dropout(decon1 ,keep_prob)
    decon1_drop = tf.reshape(decon1_drop ,(batch_size,784))
    return decon1_drop

X_encod = enconv(X , weight,bisases ,keep_prob,batch_size)

X_decod = deconv(X_encod ,weight,bisases ,keep_prob,batch_size)


cost = tf.reduce_mean(tf.pow(X - X_decod,2))


optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()

mnist = input_data.read_data_sets(r'E:\python\mnist_data', one_hot=True)
saver =  tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    total = int(mnist.train.num_examples / batch_size)
    for i in range(train_iter):
        batch_x , _ = mnist.train.next_batch(batch_size)
        for j in range(total):
            cost_ , _ = sess.run([cost,optimizer],
                                 feed_dict={X:batch_x,keep_prob:0.8})
        if i % 2 == 0:
            print("Epoch:", '%04d' % (i+1),
                  "cost=", "{:.9f}".format(cost_))
    
    print("Optimization Finished!")
    saver.save(sess,'./denoise_auto_encoder.ckpt')
    encode_decode = sess.run(
        X_decod, feed_dict={X: mnist.test.images[:batch_size],keep_prob:1.0})
    
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    print(encode_decode.shape)
    for i in range(10):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    f.show()
    plt.draw()
    plt.waitforbuttonpress()

#==============================================================================
# with tf.Session() as sess:
#     sess.run(init)
#     saver.restore(sess,'./denoise_auto_encoder.ckpt')    
#     
#     encode_decode = sess.run(
#         X_decod, feed_dict={X: mnist.test.images[:batch_size],keep_prob:1.0})
#     
#     f, a = plt.subplots(2, 10, figsize=(10, 2))
#     print(encode_decode.shape)
#     for i in range(10):
#         a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
#         a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
#     f.show()
#     plt.draw()
#     plt.waitforbuttonpress()
#==============================================================================
            
            

    
    
    
    


            