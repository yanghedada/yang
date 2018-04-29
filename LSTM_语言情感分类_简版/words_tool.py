# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 09:48:38 2018

@author: yanghe
"""

import numpy as np
import csv

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y
def read_glove_vecs(glove_file):
    with open(glove_file, 'r',encoding='utf-8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map
    
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

def sentence_to_avg(sentence, word_to_vec_map):
    words = [i.lower() for i in sentence.split()]
    avg = np.zeros((50,))
    for w in words:
        avg += word_to_vec_map[w]
    avg = avg / len(words)
    return avg
    
def sentences_to_indices(X, word_to_index, max_len):
    m = X.shape[0] 
    X_indices = np.zeros((m, max_len))
    for i in range(m):                          
        sentence_words = [w.lower() for w in X[i].split()]
        j = 0
        for w in sentence_words:
            X_indices[i, j] = word_to_index[w]
            j += 1
    return X_indices
    
def read_csv(filename = 'data/emojify_data.csv'):
    phrase = []
    emoji = []

    with open (filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)

        for row in csvReader:
            phrase.append(row[0])
            emoji.append(row[1])

    X = np.asarray(phrase)
    Y = np.asarray(emoji, dtype=int)

    return X, Y   
emoji_dictionary = {"0": "heart,good",    # :heart: prints a black instead of red heart depending on the font
                    "1": ":baseball:",
                    "2": ":simell",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:"}

def label_to_emoji(label):
    return emoji_dictionary[str(label)]