# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 16:15:57 2018

@author: yanghe
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

input_node = 784
output_node = 10

image_size = 28
num_channels = 1
num_labels = 10

conv1_deep = 32
conv1_size = 5

conv2_deep = 32
conv2_size = 5

fc_size = 512

batch_size = 128

learning_rate_base = 0.05
regulariztion_rate = 0.0001
learing_rate_decay = 0.99
training_steps = 5000
moving_average_decay = 0.99

def get_weight_l2(shape ,name, regulariztion_rate=None):
    weight = tf.get_variable('weight_%s'%str(name), shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1) )
    bias =  tf.get_variable('bias_%s'%str(name), shape=[shape[-1]], initializer=tf.constant_initializer(0.1) )
    if regulariztion_rate != None  :
        tf.add_to_collection('loss' ,tf.contrib.layers.l2_regularizer(regulariztion_rate)(weight))
    return bias , weight

def batch_norm(x ,on_train,name_scope, global_step,epsilon=1e-3, decay=0.99):  
    with tf.variable_scope(name_scope) :
        axis = list(range(len(x.get_shape()) - 1))
        size = x.get_shape().as_list()[-1]  
        scale = tf.get_variable("size%s"%(name_scope), [size], initializer=tf.constant_initializer(0.1))
        offset = tf.get_variable("offset%s"%(name_scope), [size], initializer=tf.constant_initializer(0.1)) 
        fc_mean, fc_var = tf.nn.moments(x, axis )
        ema = tf.train.ExponentialMovingAverage(moving_average_decay,global_step)
        def mean_var_with_update():
            ema_apply_op = ema.apply([fc_mean, fc_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(fc_mean), tf.identity(fc_var)
        mean, var = tf.cond(on_train,  mean_var_with_update,lambda: (ema.average(fc_mean),ema.average(fc_var)))
        return tf.nn.batch_normalization(x, mean, var, offset, scale, epsilon) 

def inference(input_tensor,dropout, on_train, regulariztion_rate, global_step):
    with tf.variable_scope('layer1_conv1'):
        conv1_biases,conv1_weight = get_weight_l2(shape=[conv1_size, conv1_size, num_channels, conv1_deep],name='layer1')
        conv1 = tf.nn.conv2d(input_tensor, conv1_weight, strides=[1,1,1,1], padding='SAME')
        conv1 = batch_norm(conv1 ,on_train,'layer1_conv1',global_step,1e-3,moving_average_decay)
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1 , conv1_biases))
        
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    with tf.variable_scope('layer3_conv2'):
        conv2_biases,conv2_weight = get_weight_l2(shape=[conv2_size, conv2_size, conv1_deep, conv2_deep],name='layer3')
        conv2 = tf.nn.conv2d(pool1, conv2_weight, strides=[1,1,1,1], padding='SAME') 
        conv1 = batch_norm(conv2 ,on_train,'layer3_conv2',global_step,1e-3,moving_average_decay)
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2 , conv2_biases))
        
    with tf.variable_scope('layer4_pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        pool_shape = pool2.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool2 ,[-1, nodes])
    
    with tf.variable_scope('layer5-fc1'):
        fc1_biases,fc1_weights = get_weight_l2([nodes, fc_size],name='layer5',regulariztion_rate=regulariztion_rate)
        fc1 = tf.matmul(reshaped, fc1_weights) + fc1_biases
        fc1 =  batch_norm(fc1 ,on_train,'layer5_fc1',global_step,1e-3,moving_average_decay)
        fc1 = tf.nn.relu(fc1)
        if dropout:
            fc1 =tf.nn.dropout(fc1, 0.5)
            
    with tf.variable_scope('layer5-fc2'):
        fc2_biases,fc2_weights = get_weight_l2([fc_size, num_labels],name='layer6',regulariztion_rate=regulariztion_rate)
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        if dropout:
            fc2 =tf.nn.dropout(fc2, 0.5)
    return fc2
            
    
def train(mnist):
    x = tf.placeholder(tf.float32, [None, image_size,  image_size, num_channels], name='x-input')
    y_ = tf.placeholder(tf.float32, [None,  num_labels], name='y-input')
    on_train = tf.placeholder(tf.bool)
    global_step = tf.Variable(0, trainable=False)
    
    y = inference(x, False,on_train, regulariztion_rate, global_step)
    
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
        validate_feed  = {x : np.reshape(mnist.validation.images[:200],(-1,image_size, image_size, num_channels)),y_:mnist.validation.labels[:200],on_train:False}
        test_feed = {x:np.reshape(mnist.test.images[:200],(-1 ,image_size, image_size, num_channels)) , y_:mnist.test.labels[:200],on_train:True}
        every_tranin = int(mnist.train.num_examples / batch_size ) 
        for i in range(training_steps):
            for j in range(every_tranin):
                bx , by = mnist.train.next_batch(batch_size)
                _ ,  step = sess.run([train_op  , global_step] , feed_dict={x:np.reshape(bx,(-1 ,image_size, image_size, num_channels)) , y_:by,on_train:True})
            #if i % 2 == 0 :
                validate_acc = sess.run(accuracy , feed_dict=validate_feed)
                print("After %d training step(s), global_step is (%s) ,validation accuracy using average model is %g " % (i, step, validate_acc))
                #saver.save(sess , 'saver/moedl1.ckpt',global_step=global_step)
        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print(("After %d training step(s), test accuracy using average model is %g" %(training_steps, test_acc)))
        
mnist = input_data.read_data_sets(r'E:\python\mnist_data', one_hot=True)
train(mnist)