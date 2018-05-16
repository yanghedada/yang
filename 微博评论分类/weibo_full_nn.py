# -*- coding: utf-8 -*-
"""
Created on Wed May 16 08:32:40 2018

@author: yanghe
"""

import numpy as np
import tensorflow as tf
import pickle
from collections import Counter
import collections
from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer


def creat_conword(pos_file, neg_file):
    vocab=[]
    
    # 把微博评论进行分词
    def porcess_file(file):
        with open(file, 'r') as f:
            vocab=[]
            lines = f.readlines()
            for line in lines:
                words = word_tokenize(line.lower())
                vocab += words
            print('last line : ',words)
            print('this weibo  word\'number is: ',len(vocab))
            return vocab
    print('porcess pos_file.......')
    vocab += porcess_file(pos_file)
    print('porcess neg_file.......')
    vocab += porcess_file(neg_file)
    
    lemmatizer = WordNetLemmatizer()
    vocab = [lemmatizer.lemmatize(word) for word in vocab] # 词形还原 (cats->cat) 
    
    word_count = Counter(vocab) # 利用counter 进行技术并生成词典｛im'the': 10120, 'a': 9444｝
    
    vocab = []
     
    # 去掉一些常用词w
    #print(word_count)
    print('com vocab_size : ',len(word_count))
    print("word size : ",len(word_count))
    for word in word_count:
        if word_count[word] < 500 and word_count[word] > 20:
            vocab.append(word)
    print('vocab_size : ', len(vocab))
    return vocab

#==============================================================================
# pos_file = 'pos.txt'
# neg_file = 'neg.txt'
# vocab = creat_conword(pos_file, neg_file)
# with open('vocab.pkl', 'wb') as f: 
#     pickle.dump(vocab, f) 
#==============================================================================
with open('vocab.pkl', 'rb') as f: 
    vocab = pickle.load(f) 

def normalize_dataset(vocab):
    dataset = []
    def string_to_vector(vocab, line, clf):
        words = word_tokenize(line.lower())
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        features = np.zeros(len(vocab))
        for word in words:
            if word in vocab:
                features[vocab.index(word)] += 1
        return list(features) + clf
    # vocab_size是 964个 ，所以每条评论 可以转换成 964 维度的样本，
    # vocab中的词出现在评论中， 相应的特征就+1， 
    max_size = []
    with open(pos_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            max_size.append(len(line))
            one_sample = string_to_vector(vocab, line,[1,0])
            dataset.append(one_sample)
    with open(neg_file, 'r') as f:
        max_size.append(len(line))
        lines = f.readlines()
        for line in lines:
            one_sample = string_to_vector(vocab, line,[0,1])
            dataset.append(one_sample)
    print('how much word in every comm max : ',max(max_size), 'min : ',min(max_size))
    print(max_size)
    return dataset

dataset = normalize_dataset(vocab)
#==============================================================================
# dataset = normalize_dataset(vocab)
# np.random.seed(100)
# dataset = np.random.permutation(dataset)  
# 
# #把整理好的数据保存到文件
# with open('dataset.pkl', 'wb') as f: 
#     pickle.dump(dataset, f) 
#==============================================================================
with open('dataset.pkl', 'rb') as f: 
    dataset = pickle.load(f)  

test_size = int(len(dataset)*0.1)

dataset = np.array(dataset)

test_dataset = dataset[:test_size]
train_dataset = dataset[test_size:]



def get_weight(shape, lambda1):
    weight = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda1)(weight))
    bias = tf.Variable(tf.constant(0.1, shape=[shape[1]]))
    return weight, bias


# 每层节点的个数

# 循环生成网络结构
def inference(x,layer_dimension):
    cur_layer = x
    n_layers = len(layer_dimension)
    in_dimension = layer_dimension[0]
    for i in range(1, n_layers):
        out_dimension = layer_dimension[i]
        weight, bias = get_weight([in_dimension, out_dimension], 0.3)
        cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
        in_dimension = layer_dimension[i]
    return cur_layer

def train(epoch,layer_dimension):
    print('train....')
    x = tf.placeholder(tf.float32, shape=(None, input_layer))
    y = tf.placeholder(tf.float32, shape=(None, out_layer))
    y_ = inference(x,layer_dimension)
    cross_entropy_mean = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1)))
    tf.add_to_collection('losses', cross_entropy_mean)
    loss = tf.add_n(tf.get_collection('losses'))

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
    #train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    #with tf.Graph().as_default() as g:
    #tf.reset_default_graph()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for epoch in range(epoch):
            i = 0
            while i < len(train_dataset):
                start = i
                end = i + batch_size
                batch_x = train_dataset[start:end][:, :-2]
                batch_y = train_dataset[start:end][:, -2:]
                _, loss_ = sess.run([train_op, loss], feed_dict={x:batch_x, y:batch_y})
                i += batch_size
                if (epoch % 2 == 0) :
                    print('准确率: ', accuracy.eval({x:test_dataset[:, :-2] , y:test_dataset[:, -2:]}))  
                
input_layer = len(vocab)
out_layer = 2
layer_dimension = [input_layer,1000,out_layer]
batch_size = 100
epoch = 1000
#train(epoch,layer_dimension)


#y= inference(x,layer_dimension)