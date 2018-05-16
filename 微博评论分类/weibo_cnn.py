# -*- coding: utf-8 -*-
"""
Created on Wed May 16 13:22:00 2018

@author: yanghe
"""

import numpy as np
import tensorflow as tf
import pickle
import collections
from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

def build_dataset(pos_file, neg_file):
    words=[]
    # 把微博评论进行分词
    def porcess_file(file):
        words = []
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                word = word_tokenize(line.lower())
                words += word
            return words
    words += porcess_file(pos_file)
    words += porcess_file(neg_file)
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

def sentences_to_indices(file, max_len,clf):
    with open(file, 'r') as f:
        lines = f.readlines()
        m = len(lines)
        X_indices = np.zeros((m, max_len+2))
        i = 0
        
        for line in lines:
            j = 0
            words = word_tokenize(line.lower())
            lemmatizer = WordNetLemmatizer()
            words = [lemmatizer.lemmatize(word) for word in words]
            for w in words:
                if w in dictionary:
                    X_indices[i,j] = dictionary[w]
                    X_indices[i,-2:] = clf 
                else :
                    X_indices[i,j] = 0
                    X_indices[i,-2:] = clf 
                if j == (max_len-1):
                    continue
                j += 1
            i += 1
    return X_indices

def dataset_build(pos_file, neg_file, max_len):
    dataset = []
    dataset.extend( sentences_to_indices(pos_file, max_len, [1, 0]))
    dataset.extend( sentences_to_indices(neg_file, max_len, [0, 1]))
    dataset = np.random.permutation(dataset)
    return dataset

vocabulary_size = 1000
#==============================================================================
# pos_file = 'pos.txt'
# neg_file = 'neg.txt'
# 
# dictionary, reverse_dictionary = build_dataset(pos_file, neg_file)
# 
#==============================================================================
#==============================================================================
# dataset = dataset_build(pos_file, neg_file, max_len=input_size)
# #把整理好的数据保存到文件
# with open('dataset_cnn.pkl', 'wb') as f: 
#     pickle.dump(dataset, f) 
#==============================================================================
with open('dataset_cnn.pkl', 'rb') as f: 
    dataset = pickle.load(f) 

test_size = int(len(dataset)*0.1)

test_dataset = dataset[:test_size]
train_dataset = dataset[test_size:]
input_size = 270
num_classes = 2
dropout_keep_prob = 0.1

#==============================================================================
# embedding_lookup(params, ids)其实就是按照ids顺序返回params中的第ids行。
# 比如说，ids=[1,3,2],就是返回params中第1,3,2行。
# 返回结果为由params的1,3,2行组成的tensor.。
#==============================================================================


def inference(x,dropout_keep_prob):
    embedding_size = 128
    embedding = tf.Variable(tf.random_normal([vocabulary_size, embedding_size]))
    embedded_chars = tf.nn.embedding_lookup(embedding, x)  
    embedded_chars_expanded = tf.expand_dims(embedded_chars, -1) 
    num_filters = 128
    filter_sizes = [3,4,5]
    pooled_outputs = []  
    for i, filter_size in enumerate(filter_sizes):
        filter_shape = [filter_size, embedding_size, 1, num_filters]  
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))  
        b = tf.Variable(tf.constant(0.1, shape=[num_filters]))  
        conv = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID")  
        h = tf.nn.relu(tf.nn.bias_add(conv, b))  
        pooled = tf.nn.max_pool(h, ksize=[1, input_size - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')  
        pooled_outputs.append(pooled)   
        filter_shape = [filter_size, embedding_size, i, num_filters] 
    num_filters_total = num_filters * len(filter_sizes)  
    h_pool = tf.concat(pooled_outputs, 3) 
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])  
    # dropout  
    h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob) 
    W_out = tf.get_variable("W", shape=[num_filters_total, num_classes], initializer=tf.contrib.layers.xavier_initializer())  
    b_out = tf.Variable(tf.constant(0.1, shape=[num_classes]))  
    output = tf.nn.xw_plus_b(h_drop, W_out, b_out)  
    return output

def train(epoch):
    print('train....')
    x = tf.placeholder(tf.int32, shape=(None, input_size))
    y = tf.placeholder(tf.float32, shape=(None, num_classes))
    y_ = inference(x,dropout_keep_prob)
    cross_entropy_mean = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_,labels=tf.argmax(y,1)))
    
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #train_op = tf.train.AdamOptimizer(0.01).minimize(cross_entropy_mean)
    train_op = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_mean)
    #with tf.Graph().as_default() as g:
    #tf.reset_default_graph()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epoch):
            i = 0
            while i < len(train_dataset):
                start = i
                end = i + batch_size
                batch_x = train_dataset[start:end][:, :-2]
                batch_y = train_dataset[start:end][:, -2:]
                _ = sess.run(train_op, feed_dict={x:batch_x, y:batch_y})
                i += batch_size
                #if (epoch % 2 == 0) :
                print('准确率: ', accuracy.eval({x:test_dataset[:, :-2] , y:test_dataset[:, -2:]})) 

batch_size = 2
epoch = 1
x = tf.placeholder(tf.int32, shape=(None, input_size))
y = tf.placeholder(tf.float32, shape=(None, num_classes))

train(epoch)