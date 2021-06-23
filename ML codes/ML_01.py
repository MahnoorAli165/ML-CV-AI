# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 18:57:05 2019

@author: Mahnoor Ali
"""
from sklearn import tree
features = [[140,1],[130,1],[150,0],[170,0]]
label = [0,0,1,1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,label)
result = clf.predict([[130,1]])
print(result)
if (result == 0):
    print ("apple")
else:
    print ("orange")