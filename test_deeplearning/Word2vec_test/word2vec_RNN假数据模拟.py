# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 12:24:01 2018

@author: yanghe
"""
import tensorflow as tf
import numpy as np

size = 2
num_layers = 3
keep_prob = tf.placeholder(tf.float32)
batch_size = 4
num_steps = 5
vocab_size = 6
max_epoch  = 1
lr = 0.01
x = np.random.rand(batch_size, num_steps)
y = np.random.rand(batch_size, num_steps)

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
    lstm_cell, output_keep_prob=keep_prob)
cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers)


initial_state = cell.zero_state(batch_size, tf.float32)
embedding = tf.get_variable("embedding", [vocab_size, size])



# input_data: [batch_size, num_steps]
# targets： [batch_size, num_steps]
input_data = tf.placeholder(tf.int32, [None, num_steps])
targets = tf.placeholder(tf.int32, [None, num_steps])


#inputs : [batch_size, num_steps,size]
inputs = tf.nn.embedding_lookup(embedding, input_data)


outputs = []

#cell_output : [size]
for time_step in range(num_steps):
    (cell_output, state) = cell(inputs[:, time_step, :], initial_state)
    outputs.append(cell_output)

output = tf.reshape(tf.concat(outputs,1), [-1, size])
softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
softmax_b = tf.get_variable("softmax_b", [vocab_size])
logits = tf.matmul(output, softmax_w) + softmax_b

#logits :[batch_size*num_steps,vocab_size]
#targets:[batch_size, num_step]

loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
    [logits],
    [tf.reshape(targets, [-1])],
    [tf.ones([batch_size * num_steps])])

final_output, _ = cell(inputs[:,1,:], state)
init = tf.global_variables_initializer()
sess =  tf.Session()


optimizer = tf.train.GradientDescentOptimizer(lr)
train_op = optimizer.minimize(loss)
sess.run(init)



for i in range(max_epoch):
    _, final_state, final_output_= sess.run([train_op, state,final_output],
                               {input_data: x,
                                targets: y,keep_prob:1.0})
print('len(final_state):',len(final_state))
print('len(final_state[0]):',len(final_state[0]))
print('final_state[0][c]:',len(final_state[0][0].shape)) 
print('final_state[1][h]',final_state[1][1].shape)
print('final_output',final_output_.shape)
  