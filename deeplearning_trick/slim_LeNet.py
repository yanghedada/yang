# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 22:55:31 2018

@author: yanghe
"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy as np

input_node = 784
output_node = 10

image_size = 28
num_channels = 1
num_labels = 10

batch_size = 128
learning_rate_base = 0.01
regulariztion_rate = 0.0001
learing_rate_decay = 0.99
training_steps = 5000
moving_average_decay = 0.99

x = tf.placeholder(tf.float32, [None, 28,  28, 1], name='x-input')
y_ = tf.placeholder(tf.float32, [None,  10], name='y-input')

def inference(x):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],  
                      activation_fn=tf.nn.relu,  
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),  
                      weights_regularizer=slim.l2_regularizer(0.0005)):  
        net = slim.conv2d(x, 32, [3,3],  1, scope='conv1')
        net = slim.max_pool2d(net, [2,2], 2, scope='pool1')
        net = slim.conv2d(net, 64, [3,3] , 1, scope='conv2')
        net = slim.max_pool2d(net, [2,2], 2, scope='pool1')
        net = slim.flatten(net)
        net = slim.fully_connected(net, 1024, scope='fc1')
        net = slim.fully_connected(net, 10, scope='fc2')
    return net
def train(mnist):
    x = tf.placeholder(tf.float32, [None, image_size,  image_size, num_channels], name='x-input')
    y_ = tf.placeholder(tf.float32, [None,  num_labels], name='y-input')
    
    y = inference(x)
    
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    
    
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y , labels=tf.argmax(y_,1))
    cross_entropy = tf.reduce_mean(cross_entropy)

    tf.add_to_collection('loss' , cross_entropy)

    regularizer_loss = tf.add_n(tf.get_collection('loss'))
    
    learning_rate = tf.train.exponential_decay(
                    learning_rate_base,
                    global_step,
                    mnist.train.num_examples / batch_size,
                    learing_rate_decay,
                    staircase=True)
    
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(regularizer_loss, global_step=global_step)
    
    with tf.control_dependencies([train_step , variables_averages_op]):
        train_op = tf.no_op(name='train')
        
    correct_prediction = tf.equal(tf.argmax(y , 1) ,tf.argmax(y_ , 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction , tf.float32))
    
    
    saver=tf.train.Saver(max_to_keep=5)
    
    with tf.Session() as sess :
        tf.global_variables_initializer().run()
        validate_feed  = {x : np.reshape(mnist.validation.images[:200],(-1,image_size, image_size, num_channels)),y_:mnist.validation.labels[:200]}
        test_feed = {x:np.reshape(mnist.test.images[:200],(-1 ,image_size, image_size, num_channels)) , y_:mnist.test.labels[:200]}
        every_tranin = int(mnist.train.num_examples / batch_size ) 
        for i in range(training_steps):
            for j in range(every_tranin):
                bx , by = mnist.train.next_batch(batch_size)
                _ ,  step = sess.run([train_op  , global_step] , feed_dict={x:np.reshape(bx,(-1 ,image_size, image_size, num_channels)) , y_:by})
            #if i % 2 == 0 :
                validate_acc = sess.run(accuracy , feed_dict=validate_feed)
                print("After %d training step(s), global_step is (%s) ,validation accuracy using average model is %g " % (i, step, validate_acc))
                #saver.save(sess , 'saver/moedl1.ckpt',global_step=global_step)
        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print(("After %d training step(s), test accuracy using average model is %g" %(training_steps, test_acc)))
        
mnist = input_data.read_data_sets(r'E:\python\mnist_data', one_hot=True)
train(mnist)      