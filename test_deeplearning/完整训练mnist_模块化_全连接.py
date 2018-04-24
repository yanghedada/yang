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
regulariztion_rate = 0.0001
learing_rate_decay = 0.99
training_steps = 5000
moving_average_decay = 0.99


def get_weight_l2(shape ,name, regulariztion_rate=None):
    weight = tf.get_variable('weight_%s'%str(name), shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1) )
    bias =  tf.get_variable('bias_%s'%str(name), shape=[shape[-1]], initializer=tf.truncated_normal_initializer(stddev=0.1) )
    if regulariztion_rate != None  :
        tf.add_to_collection('loss' ,tf.contrib.layers.l2_regularizer(regulariztion_rate)(weight))
    return bias , weight

def inference(input_tensor , regulariztion_rate):
    with tf.variable_scope('layer1'):
        biases1 , weights1 = get_weight_l2([input_node ,layer1_node] ,name=1, regulariztion_rate = regulariztion_rate)
        layer1 = tf.nn.relu(tf.matmul(input_tensor , weights1) +   biases1)
            
    with tf.variable_scope('layer2'):
        biases2 , weights2 = get_weight_l2([layer1_node ,out_node], name=2, regulariztion_rate = regulariztion_rate)
        layer2 = tf.matmul(layer1 , weights2 ) + biases2

    return layer2

def train(mnist):
    
    x = tf.placeholder(tf.float32, [None , input_node],name='x-input')
    y_ = tf.placeholder(tf.float32 , [None ,out_node] , name='y-input')
    
    # 计算不含滑动平均类的前向传播结果
    y = inference(x ,regulariztion_rate )
    
    # 定义训练轮数及相关的滑动平均类 
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
        validate_feed  = {x : mnist.validation.images[:200],y_:mnist.validation.labels[:200]}
        test_feed = {x:mnist.test.images[:200] , y_:mnist.test.labels[:200]}
        every_tranin = int(mnist.train.num_examples / batch_size ) 
        for i in range(training_steps):
            for j in range(every_tranin):
                bx , by = mnist.train.next_batch(batch_size)
                _ , step = sess.run([train_op ,  global_step] , feed_dict={x:bx , y_:by})
            if i % 2 == 0 :
                validate_acc = sess.run(accuracy , feed_dict=validate_feed)
                print("After %d training step(s), global_step is (%s) ,validation accuracy using average model is %g " % (i, step, validate_acc))
                saver.save(sess , 'saver/moedl1.ckpt',global_step=global_step)
        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print(("After %d training step(s), test accuracy using average model is %g" %(training_steps, test_acc)))
        
def evaluate(mnist):
    x = tf.placeholder(tf.float32, [None , input_node],name='x-input')
    y_ = tf.placeholder(tf.float32 , [None ,out_node] , name='y-input')
    
    validate_feed  = {x : mnist.validation.images[:200],y_:mnist.validation.labels[:200]}
    y = inference(x ,None )
    
    correct_prediction = tf.equal(tf.argmax(y , 1) ,tf.argmax(y_ , 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction , tf.float32))
    
    variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('saver/')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess , ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
            print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
        else:
            print('No checkpoint file found')
            return
    
mnist = input_data.read_data_sets(r'E:\python\mnist_data', one_hot=True)
#train(mnist)            
evaluate(mnist)      
