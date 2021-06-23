# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:02:19 2019

@author: Mahnoor Ali
"""
import numpy as np
#Input array
X=np.array([[1,1,1],
           [1,-1,1],
           [1,1,-1],
           [1,-1,-1]])

# Output array
y=np.array([1,-1,-1,-1])

#print(y.shape[0])
learnRate=0.1
print("Shape of X is: ",X.shape)

#weight=np.random.uniform(size=(np.array(X.shape[1])))
weight=np.array([0.1,0.1,0.1])
print( "weight is: ",weight)
print("First value of X is: ",X[0])
print("Trasnpose of matrix weight is: ",weight.T.shape,"\n")#giving transpose of this matrix

def activationFunc(x):#Takes some dot product as parameter
    return 1 if x >= 0 else -1

def predict(x):
    net=np.dot(x,weight)#dot product of weights and inputs
    print("Dot product of x and weight is: ",net)
    out = activationFunc(net)
    print ("Predicted Output = ",out)
    return out

for epoch in range(0,3):#3 epochs
    print('----------\nEPOCH ',epoch,'\n----------\n')
    for i in range(len(y)):
        print ("BATCH: ",i)
        print("\nInput X at ",epoch," is: ",X[i])
        print("Actual Output y at ",epoch," is: ",y[i])
        x=X[i]
        tOut=y[i]
        pOut =predict(x)
        
        error = tOut - pOut
        print("Error: ",error)
        for j in range(len(x)):
            print('\nIteration ',j,':\n')
            if (pOut!=tOut):
                new_weight = weight[j] + learnRate * (error * x[j])
                print('New weight is: ',new_weight)
                print ("--")
            else:
                print ("--Do Nothing--")

 