# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 22:41:55 2019

@author: Mahnoor Ali
"""

import pandas as pd
#from sklearn import linear_model
from sklearn import tree
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'C:\Users\Mahnoor Ali\Downloads\heart.csv')
#1-) training using linear model module
#label = df['target']
#features = df.drop('target',axis=1)
#regr = linear_model.LinearRegression()
#regr.fit(features, label)
#print(regr.predict([[65,0,0,150,225,0,0,114,0,1,1,3,3]]).tolist())
#plt.scatter(df['cp'],df['trestbps'])
#print(df['target'])

#2-) training dataset using tree
label = df['target']
features = df.drop('target',axis=1)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,label)
result = clf.predict([[63,1,3,145,233,1,0,150,0,2.3,0,0,1]])
print('Prediction:',result)
if(result==0):
    print("Not a heart patient")
else:
    print("Heart Patient")

#testing data
x= features
y= label
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=.5)
def train(clf,features,labels):
    return clf.fit(features,labels)
def predict(clf,features):
    return clf.predict(features)



my_classifier = tree.DecisionTreeClassifier()
my_classifier = train(my_classifier,x_train,y_train)
predictions = predict(my_classifier,x_test)

from sklearn.metrics import accuracy_score
print('Accuracy of Dataset',accuracy_score(y_test,predictions))


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)
print('Accuracy of LDA classifier on training set: {:.2f}'
     .format(lda.score(x_train, y_train)))
print('Accuracy of LDA classifier on test set: {:.2f}'
     .format(lda.score(x_test, y_test)))


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(x_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(x_test, y_test)))