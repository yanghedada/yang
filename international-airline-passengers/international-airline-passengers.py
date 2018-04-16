# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 23:24:06 2018

@author: Administrator
"""

import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# load the dataset
dataframe = read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
# 将整型变为float
dataset = dataset.astype('float32')
print(dataset.shape)
#==============================================================================
# plt.plot(dataset)
# plt.show()
#==============================================================================

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

# fix random seed for reproducibility
numpy.random.seed(7)

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)


# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

print(train.shape, test.shape )
# use this function to prepare the train and test datasets for modeling
look_back = 40
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

#==============================================================================
# print(trainX.shape, trainY.shape, testX.shape, testY.shape )
# print(trainX[0:10], trainY[0:10])
#==============================================================================
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
print(trainX.shape,testX.shape)
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(100, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=30, batch_size=1, verbose=2)

# make predictions
trainPredict1 = model.predict(trainX)
testPredict1 = model.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict1)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict1)
testY = scaler.inverse_transform([testY])
print(trainPredict.shape)

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
print(trainPredictPlot.shape)
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
pred = []
test1 = trainX[0:1,:,:]
for i in range(200):
    test1_pred = model.predict(test1)
    pred.append(test1_pred[0])
    test1 = numpy.reshape(test1,[-1])
    test1 = list(test1)
    test1.append(test1_pred[0])
    test1 = test1[1:]
    test1 = numpy.reshape(test1, (1, 1, -1))

pred = numpy.reshape(pred,[200,1]) 
pred = scaler.inverse_transform(pred)
test1PredictPlot = numpy.zeros([240,1])
test1PredictPlot[:, :] = numpy.nan
test1PredictPlot[look_back:len(pred)+look_back, :] = pred
print(test1PredictPlot.shape)
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
#==============================================================================
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
#==============================================================================
plt.plot(test1PredictPlot[0:180])
#
plt.show()
