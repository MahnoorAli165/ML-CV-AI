# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from sklearn import tree
features = [[140,1], [130,1], [150,0],[170,0]]
labels = [0,0,1,1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
result = clf.predict([[149, 0]])
if (result == 0):
    print ("apple")
else:
    print ("orange")