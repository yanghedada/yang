# -*- coding: utf-8 -*-
"""
Created on Thu May 17 08:10:55 2018

@author: yanghe
"""

import collections
import numpy as np
import tensorflow as tf

poetry_file = 'poetry.txt'

def read_poetry(file):
    poetrys = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            try :
                titile, content = line.strip().split(':')
                content = content.replace(' ', '')
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content:  
                    continue
                if len(content) < 5 or len(content) > 79 :
                    continue
                content = '[' + content + ']'
                poetrys.append(content)
            except Exception as e:
                pass
    return poetrys

poetrys = read_poetry(poetry_file)

def word_to_vector_and_vector_to_word(poetrys):

    print('number poetrys', len(poetrys))
    
    all_words = []
    
    # 统计汉字的出现次数，并逆序排列 
    for poetry in poetrys:
        all_words += [word for word in poetry]
    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)
    
    words = words[:len(words)] + (' ',)
    
    word_to_num = dict(zip(words, range(len(words))))
    num_to_word = dict(zip(word_to_num.values(), word_to_num.keys()))
    to_num = lambda word: word_to_num.get(word, len(words))  
    poetrys_vector = [ list(map(to_num, poetry)) for poetry in poetrys] 
    
    return poetrys_vector, num_to_word, word_to_num

poetrys_vector, num_to_word, word_to_num = word_to_vector_and_vector_to_word(poetrys)

batch_size = 128

def bulid_dataset(poetrys_vector) :
    n_chunk = len(poetrys_vector) // batch_size  
    x_batches = []  
    y_batches = [] 
    for i in range(n_chunk):  
        start_index = i * batch_size  
        end_index = start_index + batch_size  
        batches = poetrys_vector[start_index:end_index]  
        length = max(map(len,batches))  
        xdata = np.full((batch_size,length), word_to_num[' '], np.int32)  
        for row in range(batch_size):  
            xdata[row,:len(batches[row])] = batches[row]  
        ydata = np.copy(xdata)  
        ydata[:,:-1] = xdata[:,1:]  
        x_batches.append(xdata)  
        y_batches.append(ydata) 
    return x_batches, y_batches
 
x_batches, y_batches  = bulid_dataset(poetrys_vector)

def neural_network(input_data, model='lstm', rnn_size=128, num_layers=2,):
    
    if model == 'rnn':  
        cell_fun = tf.nn.rnn_cell.BasicRNNCell  
    elif model == 'gru':  
        cell_fun = tf.nn.rnn_cell.GRUCell  
    elif model == 'lstm':  
        cell_fun = tf.nn.rnn_cell.BasicLSTMCell  
   
    cell = cell_fun(rnn_size, state_is_tuple=True)  
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)  
    # batch_size 为 每首诗的长度 = 81
    initial_state = cell.zero_state(batch_size, tf.float32)  
   
    with tf.variable_scope('rnnlm'):  
        softmax_w = tf.get_variable("softmax_w", [rnn_size, 6110+1])  
        softmax_b = tf.get_variable("softmax_b", [6110+1])  
        with tf.device("/cpu:0"):  
            #vocab 为字典中词的个数。
            vocab = 6110 
            embedding = tf.get_variable("embedding", [vocab+1, rnn_size])  
            inputs = tf.nn.embedding_lookup(embedding, input_data)  
   
    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')  
    output = tf.reshape(outputs,[-1, rnn_size])  
   
    logits = tf.matmul(output, softmax_w) + softmax_b  
    probs = tf.nn.softmax(logits)  
    return logits, last_state, probs, cell, initial_state  

def train_neural_network():  
    input_data = tf.placeholder(tf.int32, [batch_size, None])  
    output_targets = tf.placeholder(tf.int32, [batch_size, None]) 
    logits, last_state, _, _, _ = neural_network(model='lstm', rnn_size=128, num_layers=2, input_data=input_data)  
    targets = tf.reshape(output_targets, [-1])  
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], 
                                                              [targets], 
                                                              [tf.ones_like(targets, dtype=tf.float32)])  
    cost = tf.reduce_mean(loss)  
    learning_rate = tf.Variable(0.0, trainable=False)  
    tvars = tf.trainable_variables()  
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)  
    optimizer = tf.train.AdamOptimizer(learning_rate)  
    train_op = optimizer.apply_gradients(zip(grads, tvars))  
   
    with tf.Session() as sess:  
        sess.run(tf.initialize_all_variables())  
   
        saver = tf.train.Saver(tf.global_variables())  
   
        for epoch in range(epochs):  
            n_chunk = len(poetrys_vector) // batch_size 
            sess.run(tf.assign(learning_rate, 0.002 * (0.97 ** epoch)))  
            n = 0  
            for batche in range(n_chunk):  
                train_loss, _ , _ = sess.run([cost, last_state, train_op], feed_dict={input_data: x_batches[n], output_targets: y_batches[n]})  
                n += 1  
            if epoch % 2 == 0:  
                print(train_loss)
                saver.save(sess , 'saver/rnn_moedl.ckpt')  
#==============================================================================
# epochs = 1
# train_neural_network()  
#==============================================================================
batch_size = 1


def gen_poetry():   
    tf.reset_default_graph()
    input_data = tf.placeholder(tf.int32, [batch_size, None]) 
    _, last_state, probs, cell, initial_state = neural_network(model='lstm', rnn_size=128, num_layers=2, input_data=input_data)   
    saver = tf.train.Saver()
    with tf.Session() as sess:  
        sess.run(tf.initialize_all_variables())  
   
        #saver.restore(sess,'./saver') 
        saver.restore(sess, tf.train.latest_checkpoint('./saver'))
        state_ = sess.run(cell.zero_state(1, tf.float32)) 
        x = np.array([list(map(word_to_num.get, '['))])   
        [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
        word = num_to_word[np.argmax(probs_)]  
        poem = ''  
        while word != ']':  
            poem += word  
            x = np.zeros((1,1))  
            x[0,0] = word_to_num[word]  
            [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})  
            word = num_to_word[np.argmax(probs_)]  
            
        return poem  
print(gen_poetry()) 

def gen_poetry_with_head(head): 
    tf.reset_default_graph()
    input_data = tf.placeholder(tf.int32, [batch_size, None])
    _, last_state, probs, cell, initial_state = neural_network()  
   
    with tf.Session() as sess:  
        sess.run(tf.initialize_all_variables())  
   
        saver = tf.train.Saver(tf.all_variables())  
        saver.restore(sess, tf.train.latest_checkpoint('./saver'))  
   
        state_ = sess.run(cell.zero_state(1, tf.float32))  
        poem = ''  
        i = 0  
        for word in head:  
            while word != '，' and word != '。':  
                poem += word  
                x = np.array([list(map(word_to_num.get, word))])  
                [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})  
                word = num_to_word[np.argmax(probs_)] 
            if i % 2 == 0:  
                poem += '，'  
            else:  
                poem += '。'  
            i += 1  
        return poem  
   
print(gen_poetry_with_head('一二三四'))  

