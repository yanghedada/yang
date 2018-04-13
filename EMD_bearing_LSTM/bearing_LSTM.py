# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 13:55:24 2018

@author: yanghe
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 19:31:49 2018

@author: yanghe
"""

import pandas as pd
import numpy as np
import cwru
import scipy.io as sio
from PyEMD import EMD
#from pyhht.emd import EMD
import pylab as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

onehot = OneHotEncoder(sparse=False)
stand = StandardScaler()

# 12DriveEndFault hanve 16 class nclasses 
# 12FanEndFault  hanve 12 class nclasses 

def load_data():
    X_train,y_train, X_test,y_test = [],[],[],[]
    for exp in ['12DriveEndFault']:
        for rpm in ['1797', '1772', '1750', '1730']:
            data = cwru.CWRU(exp, rpm, 2500)
            X_test.extend(data.X_train)
            y_test.extend(data.y_train)
            X_train.extend(data.X_test)
            y_train.extend(data.y_test)
    return np.array(X_train),np.array(y_train), np.array(X_test),np.array(y_test)
    
def data_to_imf(Data,t):
    imf = []
    for data in Data:
        #print(len(t),len(data))
        #imf_ = EMD(data).decompose()[:10]
        imf_ = EMD().emd(data,t)[:10]
        imf.append(imf_)
    return np.array(imf).reshape(-1,10,2500)

#==============================================================================
# X_train,y_train, X_test,y_test = load_data() 
# 
# 
# 
# 
# t = np.linspace(0, 1, 12000)[:2500]
# X_train_data = data_to_imf(X_train,t)
# 
# X_train_data = np.transpose(X_train_data,(0,2,1))
# 
# y_train = onehot.fit_transform(y_train.reshape(-1,1))
# 
# print(X_train_data.shape,y_train.shape)
# 
#==============================================================================

#==============================================================================
# load_data = sio.loadmat(load_fn)
# 
# S = load_data['X107_FE_time'][:120000].reshape(-1,5000)
# 
#==============================================================================
#t = np.linspace(0, 1, 12000)[:5000]





from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D, Flatten
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam    
    

    
def model(input_shape,classes):
    
    X_input = Input(shape=input_shape)
    X = Conv1D(32,kernel_size=15,strides=3,padding='valid',)(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Dropout(0.8)(X)
    
    X = GRU(units=64,return_sequences=True)(X)
    X = Dropout(0.8)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.8)(X)

    X = GRU(units=64,return_sequences=True)(X)
    X = Dropout(0.8)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.8)(X)      
    
    X = TimeDistributed(Dense(1,activation='sigmoid'))(X) 
    
    X = Flatten()(X)
    
    X = Dense(classes)(X)
    X = BatchNormalization()(X)
    X = Activation('softmax')(X)
    X = Dropout(0.8)(X)
    model = Model(inputs=X_input , outputs=X)
    return model

model = model(input_shape=(5000, 10),classes=16)

model.summary()


opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

#test_data you model is right !!!!!
X_train, y_train = np.random.randn(10,5000,10) , np.random.randn(10,16) 
model.fit(X_train, y_train, batch_size = 3, epochs=1)
#model = load_model('./models/tr_model.h5')
#loss, acc = model.evaluate(X_dev, Y_dev)
#print("Dev set accuracy = ", acc)



