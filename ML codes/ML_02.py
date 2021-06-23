# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 20:36:18 2019

@author: Mahnoor Ali
"""
import numpy as np
from sklearn import tree
from sklearn import datasets
iris= datasets.load_iris()
#print(iris.feature_names)
#print(iris.target_names)
#print(iris.data[0])
#print(iris.target[0])
#for i in range(len(iris.target)):
#    print('Example %d: label %s, features %s' %(i,iris.target[i],iris.data[i]))

#training data
#test_idx =[0,50,100]
#train_target =np.delete(iris.target,test_idx)
#train_data = np.delete(iris.data,test_idx,axis=0)
#
##testing data
#test_target= iris.target(test_idx)
#test_data = iris.data(test_idx)
#
#clf = tree.DecisionTreeClassifier()
#clf.fit(train_data,train_target)
#print(test_target)
#
#print(clf.predict(test_data))
#
##viz code
#from sklearn.externals.six import StringIO
#import pydot
#dot_data= StringIO()
#tree.export_graphviz(clf,out_file=dot_data,feature_names=iris.feature_names,
#                     class_names=iris.target_names,filled=True,rounded=True,impurity=False)
#graph= pydot.graph_from_dot_data(dot_data.getvalue())
#graph.write_pdf("iris.pdf")

x = iris.data
y = iris.target
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=.5) 

def train(clf,features,labels):
    return clf.fit(features,labels)
def predict(clf,features):
    return clf.predict(features)


my_classifier = tree.DecisionTreeClassifier()
my_classifier = train(my_classifier,x_train,y_train)
predictions = predict(my_classifier,x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))







