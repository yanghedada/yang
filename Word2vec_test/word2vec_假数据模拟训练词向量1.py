# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 20:05:18 2018

@author: yanghe
"""

import tensorflow as tf
import numpy as np

lr = 0.1
embedding_size = 3
num_samples = 4
batch_size = 5
vocabulary_size = 6
max_epoch = 100


x = np.arange(batch_size)
labels = np.arange(batch_size).reshape(-1,1)


embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], 0.0, 1.0))



weights = tf.Variable(
  tf.truncated_normal([vocabulary_size, embedding_size],
                      stddev=1.0 / np.sqrt(embedding_size)))
biases = tf.Variable(tf.zeros([vocabulary_size]))

input_data = tf.placeholder(tf.int32, [batch_size])
targets = tf.placeholder(tf.int32, [batch_size, 1])
inputs = tf.nn.embedding_lookup(embeddings, input_data)

loss = tf.reduce_mean(tf.nn.nce_loss(weights=weights,
                                     biases=biases,
                                     labels=targets,
                                     inputs=inputs,
                                     num_sampled=num_samples,
                                     num_classes=vocabulary_size
                                     ))
        
        
optimizer = tf.train.GradientDescentOptimizer(lr)
train_op = optimizer.minimize(loss)

norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),1,keep_dims=True))
normalized_embeddings = embeddings / norm

valid_embedings = tf.nn.embedding_lookup(normalized_embeddings, x[[1,2]])


init = tf.global_variables_initializer()

sess =  tf.Session()    
sess.run(init)


for i in range(max_epoch):
    _, valid_vect,embedding_vectors= sess.run([train_op, 
                                               valid_embedings,
                                               embeddings],
                               {input_data: x,targets: labels})
print('2 words shape , 1th,2th',valid_embedings.shape)
print('all embedding shape',embedding_vectors.shape)

  