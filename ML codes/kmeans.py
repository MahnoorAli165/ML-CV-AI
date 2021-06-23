# -*- coding: utf-8 -*-
"""
Created on Mon May 20 00:16:51 2019

@author: Mahnoor Ali
"""

from sklearn import cluster,datasets

import math
import numpy as np 
iris = datasets.load_iris()
X= iris.data[:,:2]

def kmeans(X,K):
    km = cluster.KMeans(n_clusters=int(K)).fit(X)
    center = km.cluster_centers_
    labels=[]
    for i in X:
        dist=[]
        for c in center:
            d= (i-c)**2
            d= np.sum(d)
            d= math.sqrt(d)
            dist.append(d)
            labels.append(dist.index(min(dist)))
    return labels
            
        
K= input("Enter number of clusters: ")
print(kmeans(X,K))