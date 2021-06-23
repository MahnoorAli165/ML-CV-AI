# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 11:47:18 2019

@author: Mahnoor Ali
"""
#hands-on python Machine Learning
#a=4;
#b=3;
#if(a==4 and b==3):
#    print("Welcome to Machine Learning")
#else:
#    print("Better Luck, Next Time"
# 

def And():
    x=[0,0,1,1]
    y=[0,1,0,1]
    print('X','Y')
    for i in range(len(x)):
        print(x[i],y[i])
    print()
    print("X.Y")
    for i in range(len(x)):
        mul= x[i]*y[i]
        print(mul)
    w1=w2=1
    b=-2
    wb=1
    bias= b*wb
    print()
    print("Summation of x and w")
    for i in range(len(x)):
        sum= x[i]+y[i]
        res=sum*(w1 or w2)
        print(res)
    print()
    print("Summation with bias")
    for i in range(len(x)):
         sum= x[i]+y[i]
         res1=sum*(w1)+bias
         print(res1)
#And()
import matplotlib.pyplot as plt
def OrThreeVariables():
    x=[0,1,0,1,0,1,0,1]
    y=[0,0,1,1,0,0,1,1]
    z=[0,0,0,0,1,1,1,1]
    print('X','Y','Z')
    for i in range(len(x)):
        print(z[i],y[i],x[i])
    print()
    print('X+Y+Z')
    for i in range(len(x)):
        add= x[i]+y[i]+z[i]
        print(add)
    w1=w2=1
    b=-1
    wb=1
    bias= b*wb
    print()
    print("Summation of x and w")
    for i in range(len(x)):
        sum= x[i]+y[i]+z[i]
        res=sum*(w1 or w2)
        print(res)
    print()
    res1=[0 for i in range(len(x))]
    print("Summation with bias")
    for i in range(len(x)):
         sum= x[i]+y[i]+z[i]
         res1[i]=sum*(w1)+bias
         print(res1[i])
    plt.plot(res1)
    plt.ylabel("Graph")
    plt.show()

OrThreeVariables()
   