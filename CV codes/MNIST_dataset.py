# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 13:15:45 2019

@author: Mahnoor Ali
"""
#1-) Import libraries and modules
import numpy as np
np.random.seed(123) # for reporducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras import backend as K
K.set_image_dim_ordering('th')


#X-train

#2-) Load pre-shufffled MNIST data into train and test sets
(X_train, y_train),(X_test,y_test)= mnist.load_data()

#3-) Preprocess input data
X_train = X_train.reshape(X_train.shape[0],1,28,28)
X_test= X_test.reshape(X_test.shape[0],1,28,28)
X_train=X_train.astype('float32')
X_test = X_test.astype('float32')
X_train/=255
X_test/=255

#4-) Preprocess class labels
Y_train = np_utils.to_categorical(y_train,10)
Y_test = np_utils.to_categorical(y_test,10)

#5-) Define model architecture
model = Sequential()

model.add(Convolution2D(32,3,3,activation='relu',input_shape=(1,28,28)))
model.add(Convolution2D(32,3,3,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

#6-) Compile Model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#7-) Fit model on training data
model.fit(X_train,Y_train,batch_size=32,nb_epoch=10,verbose=1)

#8-) Evaluate model on test data
score= model.evaluate(X_test,Y_test, verbose=0)
