# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 03:17:25 2019

@author: Mahnoor Ali
"""
    
#https://www.geeksforgeeks.org/python-image-classification-using-keras/

#1-) Import libraries and modules
import glob
import cv2
import numpy as np
np.random.seed(123) # for reporducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
#from keras.utils import np_utils
#from keras.datasets import mnist
from keras import backend as K
K.set_image_dim_ordering('th')
X_train=[]
Y_train=[]
X_test=[]
Y_test=[]
row=100
for img in glob.glob("C:/Users/Mahnoor Ali/Downloads/Semantic dataset100/image/*.jpg"):
    X_train.append(cv2.resize(cv2.imread(img),(row,row)))

for img in glob.glob("C:/Users/Mahnoor Ali/Downloads/Semantic dataset100/ground-truth/*.png"):
    Y_train.append(cv2.resize(cv2.imread(img,0),(row,row)))
    
for img in glob.glob("C:/Users\Mahnoor Ali/Downloads/Semantic dataset100/test/x_test/*.jpg"):
    X_test.append(cv2.resize(cv2.imread(img),(row,row)))

for img in glob.glob("C:/Users/Mahnoor Ali/Downloads/Semantic dataset100/test/y_test/*.png"):
    Y_test.append(cv2.resize(cv2.imread(img,0),(row,row)))  
    
#3-) Preprocess input data
X_train = np.array(X_train).astype("float32")
Y_train = np.array(Y_train).astype("float32")
X_train/=255
Y_train/=255
X_train=X_train.reshape(X_train.shape[0],3,row,row)
Y_train=Y_train.reshape(Y_train.shape[0],row*row)


X_test = np.array(X_train).astype("float32")
Y_test = np.array(Y_train).astype("float32")
X_test/=255
Y_test/=255
X_test= X_test.reshape(X_test.shape[0],3,row,row)
Y_test= Y_test.reshape(Y_test.shape[0],row*row)


#4-) Preprocess class labels
#Y_train = np_utils.to_categorical(y_train,10)
#Y_test = np_utils.to_categorical(y_test,10)

#5-) Define model architecture
model = Sequential()

model.add(Convolution2D(32,3,3,activation='relu',input_shape=(3,100,100)))
model.add(Convolution2D(32,3,3,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10000,activation='sigmoid'))

#6-) Compile Model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#7-) Fit model on training data
model.fit(X_train,Y_train,batch_size=25,nb_epoch=4,verbose=1)

#8-) Evaluate model on test data
score= model.evaluate(X_test,Y_test, verbose=0)
print (score)
