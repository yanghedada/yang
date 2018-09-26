# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 08:43:37 2018

@author: yanghe
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 20:39:38 2018

@author: yanghe
"""

import tensorflow as tf
import numpy as np
from  words_tool import *



X_train, Y_train = read_csv('data/train_emoji.csv')
X_test, Y_test = read_csv('data/tesss.csv')
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')
X_train_indices = sentences_to_indices(X_train, word_to_index, 10)
Y_train_oh = convert_to_one_hot(Y_train, C = 5)
X_test_indices = sentences_to_indices(X_test, word_to_index, 10)
Y_test_oh = convert_to_one_hot(Y_test, C = 5)


num_steps = 10
size = 50
n_hidden = 128
batch_size = 4
vocab_size = 400000
max_epoch  = 10
learning_rate = 0.01
n_classes = 5
count = 0

input_data = tf.placeholder(tf.int32, [None, num_steps])
targets = tf.placeholder(tf.int32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

def model(input_data,on_training):
    with tf.variable_scope("embed") :
        emb_matrix = np.zeros((vocab_size+1, size))
        for  word, index in word_to_index.items():
            emb_matrix[index, :] = word_to_vec_map[word]
        embedding = tf.get_variable("embedding", initializer=tf.constant(emb_matrix,tf.float32),trainable=False)
        inputs = tf.nn.embedding_lookup(embedding , input_data)
    
    with tf.variable_scope('Bi_RNN'):
        inputs = tf.transpose(inputs, [1, 0, 2])
        inputs = tf.reshape(inputs, [-1, size])
        inputs = tf.split(inputs, num_steps)
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
        
        if on_training: 
            lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell,output_keep_prob=keep_prob)
            lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell,output_keep_prob=keep_prob)
            
        outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell,
                                                                lstm_bw_cell,
                                                                inputs,
                                                                dtype=tf.float32)
    with tf.name_scope('softmx'):
        weights = tf.get_variable('weights', shape=[2*n_hidden, n_classes], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', shape=[n_classes], initializer=tf.truncated_normal_initializer(stddev=0.1))
    outputs = tf.matmul(outputs[-1], weights) + biases 
    return outputs                
def train():
    on_training = True
    with tf.variable_scope("pred"):
        pred = model(input_data,on_training)
    saver = tf.train.Saver()
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=targets))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    correct_pred = tf.equal(tf.argmax(targets, 1), tf.argmax(pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    with tf.Session() as sess :
        tf.global_variables_initializer().run()
        for i in range(max_epoch):
            _= sess.run([train_op ],feed_dict={input_data: X_train_indices, 
                                               targets: Y_train_oh,
                                               keep_prob:0.8})
            if i % 5 == 0 :
                accuracy_= sess.run(accuracy,feed_dict={input_data:X_test_indices, 
                                                        targets:Y_test_oh,
                                                        keep_prob:1.0})
            print("After %d , validation accuracy is %s " % (i,accuracy_))
        saver.save(sess , 'saver/moedl_em.ckpt')

    
def predict():
    global count
    on_training = False
    with tf.variable_scope("pred") as scope:
        if count == 0:
            pass
        else:
            scope.reuse_variables()
        pred = model(input_data,on_training)
        count += 1
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver.restore(sess,'./saver/moedl_em.ckpt')
        usr_input = input("Write the beginning of your want to say , but not benyond 10 words :")
        X_test = sentences_to_indices(np.array([usr_input]), word_to_index, 10)
        predict_= sess.run(pred,feed_dict={input_data:X_test,keep_prob:1.0})
        print('you input is ',usr_input,'machine predict you want tp say:--->',label_to_emoji(np.argmax(predict_)))
        
#train()
predict()
#I love taking breaks
# This girl is messing with me
# she got me a nice present
# work is hard
