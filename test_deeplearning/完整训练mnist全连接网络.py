# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 10:16:51 2018

@author: yanghe
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data




input_node = 784
out_node = 10
layer1_node = 500

batch_size = 128

learning_rate_base = 0.8
regulariztion_rate = 0.001
learing_rate_decay = 0.99
training_steps = 5000
moving_average_decay = 0.99

def get_weight_l2(shape ,name, l2=False ):
    weight = tf.Variable(tf.truncated_normal(shape , stddev=0.1),dtype=tf.float32 , name='weight%s'%(str(name)))
    if l2  :
        tf.add_to_collection('loss' ,tf.contrib.layers.l2_regularizer(regulariztion_rate)(weight))
    bias = tf.Variable(tf.random_normal([shape[-1]]))
    return bias , weight

def inference(input_tensor , avg_class ,  weights1, biases1, weights2, biases2):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor , weights1) +   biases1)
        return tf.matmul(layer1 , weights2 ) + biases2
    else :
        layer1 = tf.nn.relu(tf.matmul(input_tensor , avg_class.average(weights1)) + biases1)
        return tf.matmul(layer1 , avg_class.average(weights2))  + biases2

def train(mnist):
    
    x = tf.placeholder(tf.float32, [None , input_node],name='x-input')
    y_ = tf.placeholder(tf.float32 , [None ,out_node] , name='y-input')
    
    biases1 , weights1 = get_weight_l2([input_node ,layer1_node] ,name=1, l2 = True)
    biases2 , weights2 = get_weight_l2([layer1_node ,out_node], name=2, l2=True)
    
    # 计算不含滑动平均类的前向传播结果
    y = inference(x , None ,  weights1, biases1, weights2, biases2)
    
    # 定义训练轮数及相关的滑动平均类 
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)
    
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
        
    correct_prediction = tf.equal(tf.argmax(average_y , 1) ,tf.argmax(y_ , 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction , tf.float32))
    
    
    
    with tf.Session() as sess :
        tf.global_variables_initializer().run()
        validate_feed  = {x : mnist.validation.images[:200],y_:mnist.validation.labels[:200]}
        test_feed = {x:mnist.test.images[:200] , y_:mnist.test.labels[:200]}
        every_tranin = int(mnist.train.num_examples / batch_size ) 
        for i in range(training_steps):
            for j in range(every_tranin):
                bx , by = mnist.train.next_batch(batch_size)
                sess.run(train_op , feed_dict={x:bx , y_:by})
            if i % 100 == 0 :
                validate_acc = sess.run(accuracy , feed_dict=validate_feed)
                print("After %d training step(s),  validation accuracy using average model is %g " % (i, validate_acc))
        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print(("After %d training step(s), test accuracy using average model is %g" %(training_steps, test_acc)))
mnist = input_data.read_data_sets(r'E:\python\mnist_data', one_hot=True)
train(mnist)            
        
