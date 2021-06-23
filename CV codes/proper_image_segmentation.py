# -*- coding: utf-8 -*-

import os
import cv2
import sampling
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np

pred = []
def prediction(model, img, frame ="pred"):
    #preprocessing
    
    global pred
    images=np.array([img])
    pred=model.predict(images)[0].reshape((200,200))*255
    pred=pred.astype("uint8")
    mask_on = cv2.bitwise_and(img, img, mask = pred)
    cv2.imshow(frame, pred)
    cv2.imshow("mask on", mask_on)
    cv2.imshow("image", img)
    cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()

x=[]
y=[]
width, height = 200, 200



load = 5000000
total_loaded = 0
for root, dirs, files in os.walk("Training Data for Dogs\\dataset\\dataset\\masks", topdown=False):
   for name in files:
      if total_loaded >= load:
          break
      found_image=False
      path = os.path.join(root, name)
      mask=cv2.imread(path, 0)
      
      
      image_path=name.split(".")
      image_name=image_path[0]
      image_name=image_name + ".JPEG"
      
      if os.path.isfile("Training Data for Dogs\\dataset\\dataset\\images\\" + image_name):
            image_path="Training Data for Dogs\\dataset\\dataset\\images\\" + image_name
            image=cv2.imread(image_path)
            found_image=True
            
    
      if(found_image):
        mask=cv2.resize(mask, (200,200), cv2.INTER_CUBIC)
        y.append(mask)
        image=cv2.resize(image, (200,200), cv2.INTER_CUBIC)
        x.append(image)
        total_loaded += 1



    


for k in range(len(y)):
    y[k]=np.reshape(y[k], (200*200))

x = np.array(x)
x=x.astype("float32")
x=x/255
y=np.array(y)
y=y.astype("float32")
y=y/255

    
X_train=x
X_test=x
Y_train=y
Y_test=y

X_train=x[0:int(len(x)/2)]
X_test=x[int(len(x)/2):len(x)]
Y_train=np.array(y[0:int(len(y)/2)])
Y_test=np.array(y[int(len(y)/2):len(y)])

model = Sequential()
model.add(Convolution2D(32, 5, 5, activation='relu', input_shape=(200, 200, 3))) 
model.add(Convolution2D(32, 3, 3, activation='relu')) 
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(32, 5, 5, activation='relu')) 
model.add(Dropout(0.5)) 
model.add(Flatten())
model.add(Dense(128, activation='relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(200*200, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



"""model.fit(X_train, Y_train, batch_size=32, nb_epoch=10, verbose=1)
score = model.evaluate(X_test, Y_test, verbose=0)
model.save("sat1")"""
model.load_weights("sat1")
for i in range(10):
    prediction(model, (X_test[i]*255).astype("uint8"), "prediction: " + str(i))


cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()
  
