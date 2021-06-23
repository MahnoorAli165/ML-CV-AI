# -*- coding: utf-8 -*-
"""
Created on Sun May 19 21:25:26 2019

@author: Mahnoor Ali
"""
import random
def equalArrays(A, B):
    if(len(A) <= len(B)):
        for i in range(len(A)):
            if(A[i] != B[i]):
                return False
        return True
    return False

def closetoCenter(distance):
    min = distance[0]
    index = 0
    for i in range(len(distance)):
        if(distance[i] <= min):
            min = distance[i]
            index = i
    return index


def newCenter(clusters, X, vectors):
    centers = []
    for cluster in clusters:
        center = []
        for i in range(len(X[0])):
            temp = 0
            count = 0
            for j in range(len(vectors)):
                if(vectors[j] == cluster):
                    temp += X[j][i]
                    count += 1
            temp = temp/count
            temp = round(temp, 2)
            center.append(temp)
        centers.append(center)
    return centers

    

def kmeans(X,K):
    cluster = []
    for i in range(K):
        cluster.append(i)
    
    center=[]
    for i in range(K):
        a = random.randint(0,len(X)-1)
        center.append(X[a])
    distances= []
    vectors =[]

    while(True):
          distances = []
          vectors = []
          for val in X:
            temp_distance = []
            for c in center:
                distance = []
                for i in range(len(c)):
                    distance.append(round(abs(c[i]-val[i]), 2))
                temp_distance.append(distance)
            distances.append(temp_distance)
    
       
          for i in range(len(distances)):
            vectors.append(closetoCenter(distances[i]))
          new_centers = newCenter(cluster, X, vectors)
          if(equalArrays(center, new_centers)):
            break
          center = new_centers.copy()
    
    return vectors

X= [[12,39], [20,36],[28,30], [18,52],[29,54],[33,46],[24,55],[45,59],
    [45,63],[52,70],[51,66],[52,63],[55,58],[53,23],[55,14],[61,8],[64,19],[69,7],[72,24]]
K=2     
a=kmeans(X,K)
print(a)
            
    
            
            
                
