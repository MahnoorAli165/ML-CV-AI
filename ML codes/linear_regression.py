# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:55:04 2019

@author: Mahnoor Ali
"""
import matplotlib.pyplot as plt
import math
def linear_regression():
    x=[65,50,55,65,55,70,65,70,55,70,50,55]
    y=[85,74,76,90,85,87,94,98,81,91,76,74]
    mul=sum1=0
    for i in range(len(x)):
        mul=x[i]*y[i]
        sum1=sum1+mul
    print("Summation(X.Y)",sum1)
    n=len(x)
    sumx=0
    for i in range(len(x)):
        sumx=sumx+x[i]
    print("Sum of X",sumx)
    sumy=0
    for i in range(len(y)):
        sumy=sumy+y[i]
    print("Sum of Y",sumy)
    sum_x_sq=0
    for i in range(len(x)):
        sq= math.pow(x[i],2)
        sum_x_sq= sum_x_sq+sq
    print("Sum of(x^2)",sum_x_sq)
    sum_x_1= math.pow(sumx,2)
    print("(Sum of x)^2",sum_x_1)
    m= (((n*sum1)-(sumx)*(sumy))/((n*sum_x_sq)-(sum_x_1)))
    print("Slope is",m)
    
    b= (((sumy)/n)-(m*(sumx/n)))
    print("b is",b)
    yp=[0 for i in range(len(y))]
    for i in range(len(x)):
        yp[i]= (m*x[i])+b
    print("Y predicted",yp)
    
    diffy=[0 for i in range(len(y))]
    for i in range(len(y)):
        diffy[i]=yp[i]-y[i]
    print("yp-y",diffy)
    
    ysq=[0 for i in range(len(y))]
    for i in range(len(y)):
        ysq[i]= math.pow(diffy[i],2)
    print("(yp-y)^2",ysq)
    ysq_sum=0
    for i in range(len(y)):
        ysq_sum=ysq_sum+ysq[i]
    print("Summation of (yp-y)^2",ysq_sum)
    
    
#    plt.plot(x,y)
    plt.scatter(x,y)
    plt.plot(x,y)
    plt.title("X and Y")
    plt.scatter(x,yp)
    plt.plot(x,yp)
#    plt.scatter()
    
#    
#    for i in range(len(y)):
#        rmse= math.sqrt((yp))
    
linear_regression()
        