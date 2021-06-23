# -*- coding: utf-8 -*-
"""
Created on Sun May 19 11:52:23 2019

@author: Sharjeel Ahmad
"""

import random

def equalArrays(A, B):
    if(len(A) <= len(B)):
        for i in range(len(A)):
            if(A[i] != B[i]):
                return False
        return True
    return False

def closestCenter(distances):
    smallest = distances[0]
    index = 0
    for i in range(len(distances)):
        if(distances[i] <= smallest):
            smallest = distances[i]
            index = i
    return index

def newCenters(clusters, X, labels):
    centers = []
    for cluster in clusters:
        center = []
        for i in range(len(X[0])):
            temp = 0
            count = 0
            for j in range(len(labels)):
                if(labels[j] == cluster):
                    temp += X[j][i]
                    count += 1
            temp = temp/count
            temp = round(temp, 2)
            center.append(temp)
        centers.append(center)
    return centers

def kmeans(X, K):
    clusters = []
    for i in range(K):
        clusters.append(i)
    
    centers = []
    
    for i in range(K):
        index = random.randint(0, len(X)-1)
        centers.append(X[index])
        
    labels = []
    distances = []
    
    while(True):
        distances = []
        labels = []
        for x in X:
            temp_distance = []
            for center in centers:
                distance = []
                for i in range(len(center)):
                    distance.append(round(abs(center[i]-x[i]), 2))
                temp_distance.append(distance)
            distances.append(temp_distance)
        
        for i in range(len(distances)):
            labels.append(closestCenter(distances[i]))
        new_centers = newCenters(clusters, X, labels)
        if(equalArrays(centers, new_centers)):
            break
        centers = new_centers.copy()
    
    return labels


X = [[5.1, 3.5],[4.9, 3. ],[4.7, 3.2],[4.6, 3.1],
 [5. , 3.6],[5.4, 3.9],[4.6, 3.4],[5. , 3.4],
 [4.4, 2.9],[4.9, 3.1],[5.4, 3.7],[4.8, 3.4],
 [4.8, 3. ],[4.3, 3. ],[5.8, 4. ],[5.7, 4.4],
 [5.4, 3.9],[5.1, 3.5],[5.7, 3.8],[5.1, 3.8],
 [5.4, 3.4],[5.1, 3.7],[4.6, 3.6],[5.1, 3.3],
 [4.8, 3.4],[5. , 3. ],[5. , 3.4],[5.2, 3.5],
 [5.2, 3.4],[4.7, 3.2],[4.8, 3.1],[5.4, 3.4],
 [5.2, 4.1],[5.5, 4.2],[4.9, 3.1],[5. , 3.2],
 [5.5, 3.5],[4.9, 3.6],[4.4, 3. ],[5.1, 3.4],
 [5. , 3.5],[4.5, 2.3],[4.4, 3.2],[5. , 3.5],
 [5.1, 3.8],[4.8, 3. ],[5.1, 3.8],[4.6, 3.2],
 [5.3, 3.7],[5. , 3.3],[7. , 3.2],[6.4, 3.2],
 [6.9, 3.1],[5.5, 2.3],[6.5, 2.8],[5.7, 2.8],
 [6.3, 3.3],[4.9, 2.4],[6.6, 2.9],[5.2, 2.7],
 [5. , 2. ],[5.9, 3. ],[6. , 2.2],[6.1, 2.9],
 [5.6, 2.9],[6.7, 3.1],[5.6, 3. ],[5.8, 2.7],
 [6.2, 2.2],[5.6, 2.5],[5.9, 3.2],[6.1, 2.8],
 [6.3, 2.5],[6.1, 2.8],[6.4, 2.9],[6.6, 3. ],
 [6.8, 2.8],[6.7, 3. ],[6. , 2.9],[5.7, 2.6],
 [5.5, 2.4],[5.5, 2.4],[5.8, 2.7],[6. , 2.7],
 [5.4, 3. ],[6. , 3.4],[6.7, 3.1],[6.3, 2.3],
 [5.6, 3. ],[5.5, 2.5],[5.5, 2.6],[6.1, 3. ],
 [5.8, 2.6],[5. , 2.3],[5.6, 2.7],[5.7, 3. ],
 [5.7, 2.9],[6.2, 2.9],[5.1, 2.5],[5.7, 2.8],
 [6.3, 3.3],[5.8, 2.7],[7.1, 3. ],[6.3, 2.9],
 [6.5, 3. ],[7.6, 3. ],[4.9, 2.5],[7.3, 2.9],
 [6.7, 2.5],[7.2, 3.6],[6.5, 3.2],[6.4, 2.7],
 [6.8, 3. ],[5.7, 2.5],[5.8, 2.8],[6.4, 3.2],
 [6.5, 3. ],[7.7, 3.8],[7.7, 2.6],[6. , 2.2],
 [6.9, 3.2],[5.6, 2.8],[7.7, 2.8],[6.3, 2.7],
 [6.7, 3.3],[7.2, 3.2],[6.2, 2.8],[6.1, 3. ],
 [6.4, 2.8],[7.2, 3. ],[7.4, 2.8],[7.9, 3.8],
 [6.4, 2.8],[6.3, 2.8],[6.1, 2.6],[7.7, 3. ],
 [6.3, 3.4],[6.4, 3.1],[6. , 3. ],[6.9, 3.1],
 [6.7, 3.1],[6.9, 3.1],[5.8, 2.7],[6.8, 3.2],
 [6.7, 3.3],[6.7, 3. ],[6.3, 2.5],[6.5, 3. ],
 [6.2, 3.4],[5.9, 3. ]]


K = 2

labels = kmeans(X, K)
print(labels)
