# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 16:40:04 2018

@author: yanghe
"""
import tensorflow as tf
import numpy as np
import io

text = io.open('shakespeare.txt', encoding='utf-8').read().lower()

chars = sorted(list(set(text)))
learning_rate = 0.01
training_steps = 30
batch_size = 64
display_step = 10
n_steps = 38
n_input = 38
n_hidden = 256
n_classes = 38
# 构建数据集 
#  X:[len(text)-n_steps，n_steps]
#
def build_data(text, n_steps = 40, stride = 3):
    X = []
    Y = []
    for i in range(0, len(text) - n_steps, stride):
        X.append(text[i: i + n_steps])
        Y.append(text[i + n_steps])
    print('number of training examples:', len(X))
    
    return X, Y


def vectorization(X, Y, n_input, char_indices, n_steps = n_steps):
    m = len(X)
    x = np.zeros((m, n_steps, n_input), dtype=np.bool)
    y = np.zeros((m, n_input), dtype=np.bool)
    for i, sentence in enumerate(X):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[Y[i]]] = 1
        
    return x, y 
    
    
def sample(preds, temperature=1.0):
    e_x = np.exp(preds - np.max(preds))
    preds = e_x / e_x.sum(axis=1)
    out = np.random.choice(range(len(chars)), p = preds.ravel())
    return out
    

    
def random_mini_batches(x,y,mini_bath_size =64,seed =0):
    np.random.seed(seed)
    m = x.shape[0]
    mini_batches = []
    permutation = list(np.random.permutation(m))
    shuffled_x = x[permutation]
    shuffled_y = y[permutation]
    
    num_complete_minibatches = int(np.floor(m/mini_bath_size))
    for k in range(0,num_complete_minibatches):
        mini_batch_x = shuffled_x[k*mini_bath_size:(k+1)*mini_bath_size,:,:]
        mini_batch_y = shuffled_y[k*mini_bath_size:(k+1)*mini_bath_size,]
        mini_batch = (mini_batch_x,mini_batch_y)
        mini_batches.append(mini_batch)
    if m % mini_bath_size  != 0:
        mini_batch_x = shuffled_x[(k+1)*mini_bath_size:,:,:]
        mini_batch_y = shuffled_y[(k+1)*mini_bath_size:,:]
        mini_batch = (mini_batch_x,mini_batch_y)
        mini_batches.append(mini_batch)
    return mini_batches

def BiRNN(x):
    with tf.variable_scope('Bi_RNN'):
        weights = tf.get_variable('weights', shape=[2*n_hidden, n_classes], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', shape=[n_classes], initializer=tf.truncated_normal_initializer(stddev=0.1))
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

def train():
    y_ = BiRNN(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    saver = tf.train.Saver()
    with tf.Session() as sess :
        tf.global_variables_initializer().run()
        mini_batches = random_mini_batches(X_, Y_, mini_bath_size=batch_size) 
        for i in range(training_steps):
            for x_train,y_train in mini_batches:
                _ = sess.run(train_op, feed_dict={x:x_train , y:y_train })
            if i % 5 == 0 :
                j = np.random.randint(len(mini_batches[0]))
                validate_acc = sess.run([accuracy], feed_dict={x:mini_batches[j][0] , y:mini_batches[j][1]})
                print("After %d , validation accuracy is %s " % (i,  validate_acc))
        test_acc=sess.run(accuracy,feed_dict={x:mini_batches[-1][0] , y:mini_batches[-1][1]})
        print(("After %d training step(s), test accuracy using average model is %.2f" %(training_steps, test_acc)))    
        saver.save(sess , 'saver/moedl1.ckpt')
    
def generate_output():
    #with tf.variable_scope('Bi_RNN') as scope:
    #scope.reuse_variables()
    y_ = BiRNN(x)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,'saver/moedl1.ckpt')
        generated = ''
        usr_input = input("Write the beginning of your poem, the Shakespeare machine will complete it. Your input is: ")
        sentence = ('{0:0>' + str(n_steps) + '}').format(usr_input).lower()
        generated += usr_input 
        for i in range(400):
            x_pred = np.zeros((1, n_steps, len(chars)))
    
            for t, char in enumerate(sentence):
                if char != '0':
                    x_pred[0, t, char_indices[char]] = 1.
            preds = sess.run(y_ ,feed_dict={x:x_pred})
    
            
            next_index = sample(preds, temperature = 1.0)
            next_char = indices_char[next_index]
            generated += next_char
            sentence = sentence[1:] + next_char
            if next_char == '\n':
                continue
        print(generated)
        
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
X, Y = build_data(text, n_steps, stride = 3)
X_, Y_ = vectorization(X, Y, n_input = len(chars), char_indices = char_indices) 

x = tf.placeholder(tf.float32, [None, n_steps,n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

#train()
generate_output()




