# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 13:00:53 2018

@author: yanghe
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(r'E:\python\mnist_data', one_hot=True)
learning_rate = 0.01
training_steps = 1000
batch_size = 128
display_step = 10

n_input = 28
n_steps = 28
n_hidden = 128
n_classes = 10

x = tf.placeholder(tf.float32, [None, n_steps*n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

def BiRNN(x):
    with tf.name_scope('Bi_RNN'):
        weights = tf.get_variable('weights', shape=[2*n_hidden, n_classes], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', shape=[n_classes], initializer=tf.truncated_normal_initializer(stddev=0.1))
        x = tf.reshape(x, [-1, 28, 28])
        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [-1, n_steps])
        x = tf.split(x, n_steps)
        
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
        outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell,
                                                                lstm_bw_cell,
                                                                x,
                                                                dtype=tf.float32)
        
    return tf.matmul(outputs[-1], weights) + biases

y_ = BiRNN(x)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


with tf.Session() as sess :
        tf.global_variables_initializer().run()
        validate_feed  = {x : mnist.validation.images[:200],y:mnist.validation.labels[:200]}
        test_feed = {x:mnist.test.images[:200] , y:mnist.test.labels[:200]}
        every_tranin = int(mnist.train.num_examples / batch_size ) 
        for i in range(training_steps):
            for j in range(every_tranin):
                bx , by = mnist.train.next_batch(batch_size)
                _ = sess.run(train_op, feed_dict={x:bx , y:by})
            #if i % 2 == 0 :
                validate_acc = sess.run([accuracy], feed_dict=validate_feed)
                print("After %d , validation accuracy is %s " % (i,  validate_acc))
        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print(("After %d training step(s), test accuracy using average model is %.2f" %(training_steps, test_acc)))    











