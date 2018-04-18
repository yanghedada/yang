# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 12:59:27 2018

@author: yanghe
"""

from sklearn import datasets 


iris = datasets.load_iris()
digits = datasets.load_digits()
#==============================================================================
# print(iris.data[0:10],iris.target[0:10])
# print(iris.data.shape)
# print(digits.data.shape)
#==============================================================================

#==============================================================================
# from sklearn import svm
# clf = svm.SVC(gamma=0.001,C=100)
# clf.fit(digits.data[:-10],digits.target[:-10])
# pred = clf.predict(digits.data[-10:])
# 
# print(pred,digits.target[-10:])
# 
#==============================================================================

#==============================================================================
# from sklearn.externals import joblib 
# #joblib.dump(clf,'filename.pkl')
# clf21 = joblib.load('filename.pkl') 
# predclf21 = clf21.predict(digits.data[-9:])
# print(predclf21,digits.target[-9:])
# 
#==============================================================================

import numpy as np
from sklearn import random_projection
#==============================================================================
# rng = np.random.RandomState(0)
# x = rng.rand(10,2000)
# x = np.array(x,dtype = np.float32)
# print(x.shape)
# print(x.dtype)
# transformer = random_projection.GaussianRandomProjection()
# x_new = transformer.fit_transform(x)
# print(x_new.dtype)
#==============================================================================

from sklearn.svm import SVC

clf = SVC()
#==============================================================================
# clf.fit(iris.data[:-20],iris.target[:-20])
# pred = clf.predict(iris.data[-10:])
# print(pred,iris.target[-10:])
# 
# print(iris.target_names[iris.target][-10:])
# clf.fit(iris.data,iris.target_names[iris.target])
# print(clf.predict(iris.data[-10:]))
#==============================================================================

#==============================================================================
# rng = np.random.RandomState(0)
# x = rng.rand(100,10)
# y = rng.binomial(1,0.5,100)
# x_test =rng.rand(5,10)
# print(x[0:5],y[0:5])
# 
# 
# clf = SVC()
# clf.set_params(kernel='linear').fit(x,y)
# pred = clf.predict(x_test)
# print(pred)
# 
# clf.set_params(kernel = 'rbf',).fit(x,y)
# pred = clf.predict(x_test)
# print(pred)
#==============================================================================


from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer

x = [[1,2],[2,4],[4,5],[3,2],[3,1]]
y = [0,0,1,1,2]
classif = OneVsRestClassifier(estimator=SVC(random_state=0))
#==============================================================================
# pred = classif.fit(x,y).predict(x)
# print (pred)
# 
# y  = LabelBinarizer().fit_transform(y)
# print(y)
# 
# pred = classif.fit(x,y).predict(x)
# print(pred)
# 
#==============================================================================

#==============================================================================
# from sklearn.preprocessing import MultiLabelBinarizer
# y = [[0,1],[0,2],[1,3],[0,2,3],[2,4]]
# y = MultiLabelBinarizer().fit_transform(y)
# print(y)
# pred = classif.fit(x,y).predict(x)
# print(pred)
#==============================================================================

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

#==============================================================================
# x = digits.data[:,1]
# y = digits.data[:,2]
# species = digits.target
# print(x.shape,species.shape,y[0:20])
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.set_title('iris datasets by pca',size=14)
# ax.scatter(digits.data[:,1],digits.data[:,2],digits.data[:,3],c=species)
# ax.set_xlabel('first eigenvector')
# ax.set_ylabel('second eigenvector')
# ax.set_zlabel('thrid eigenvector')
# ax.w_xaxis.set_ticklabels(())
# ax.w_yaxis.set_ticklabels(())
# ax.w_zaxis.set_ticklabels(())
# ax.view_init(elev=100,azim=0)
# 
#==============================================================================

#==============================================================================
# x = iris.data[:,1]
# y = iris.data[:,2]
# species = iris.target
# x_reduce = PCA(n_components=3).fit_transform(iris.data)
# print(x_reduce.shape)
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.set_title('iris datasets by pca',size=14)
# ax.scatter(x_reduce[:,0],x_reduce[:,1],x_reduce[:,2],c=species)
# ax.set_xlabel('first eigenvector')
# ax.set_ylabel('second eigenvector')
# ax.set_zlabel('thrid eigenvector')
# ax.w_xaxis.set_ticklabels(())
# ax.w_yaxis.set_ticklabels(())
# ax.w_zaxis.set_ticklabels(())
# ax.view_init(elev=100,azim=0)
#==============================================================================


#==============================================================================
# import numpy as np 
# from sklearn.neighbors import KNeighborsClassifier
# from matplotlib.colors import ListedColormap
# knn = KNeighborsClassifier()
# np.random.seed(0)
# 
# x = iris.data[:,:2]
# y = iris.target
# x_min , x_max = x[:,0].min() -0.5,x[:,0].max() + 0.5
# y_min , y_max = x[:,1].min() -0.5,x[:,1].max() + 0.5
# cmap_light = ListedColormap(['#AAAAFF','#AAFFAA','#FFAAAA'])
# h = 0.2
# xx , yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
# 
# knn.fit(x,y)
# z = knn.predict(np.c_[xx.ravel(),yy.ravel()])
# print(z.shape,xx.shape)
# z = z.reshape(xx.shape)
# print(z[0:10])
# plt.figure()
# plt.pcolormesh(xx,yy,z,cmap = cmap_light)
# plt.scatter(x[:,0],x[:,1],c=y)
# plt.xlim([x_min,x_max])
# plt.ylim(y_min,y_max)
#==============================================================================






















