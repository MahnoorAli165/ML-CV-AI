# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 12:22:19 2019

@author: Mahnoor Ali
"""
import math
mat=[[20,20,20,20,165,165,165,165],
     [20,20,20,20,165,165,165,165],
     [20,20,20,20,165,165,165,165],
     [20,20,20,20,165,165,165,165],
     [20,20,20,20,165,165,165,165],
     [20,20,20,20,165,165,165,165],
     [20,20,20,20,165,165,165,165]]

fil =[[1,0,-1],
      [1,0,-1],
      [1,0,-1]]

row = (int)(len(fil)/2)
col = (int)(len(fil[0])/2)
b=[[0 for x in range(len(mat[0]))]for y in range(len(mat))]
for i in range(row,len(mat)-row):
    for j in range(col,len(mat[i])-col):
        sum = 0 
        for k in range(len(fil)):
            for l in range(len(fil[k])):
                sum = sum + (mat[(i-row)+k][(j-col)+l]*fil[k][l])
        b[i][j]=sum
print("Applying Horizontal Gradient Filter")
for i in range(len(b)):
    for j in range(len(b[i])):
          print('%4d'%b[i][j],end=" ")
    print()
print()
fil1 =[[1,1,1],
      [0,0,0],
      [-1,-1,-1]]
row1 = (int)(len(fil1)/2)
col1 = (int)(len(fil1[0])/2)
c=[[0 for x in range(len(mat[0]))]for y in range(len(mat))]
for i in range(row,len(mat)-row1):
    for j in range(col,len(mat[i])-col1):
        sum1 = 0 
        for k in range(len(fil1)):
            for l in range(len(fil1[k])):
                sum1 = sum1 + (mat[(i-row1)+k][(j-col1)+l]*fil1[k][l])
        c[i][j]=sum1
print("Applying Vertical Gradient Filter")        
for i in range(len(c)):
    for j in range(len(c[i])):
          print('%4d'%c[i][j],end=" ")
    print()     
print()
d=[[0 for x in range(len(mat[0]))]for y in range(len(mat))]
for i in range(len(d)):
    for j in range(len(d[i])):
        d[i][j] = math.sqrt(((math.pow(b[i][j],2))+(math.pow(c[i][j],2))))

print("Resultant Gradient Magnitude")
for i in range(len(d)):
    for j in range(len(d[i])):
          print('%4d'%d[i][j],end=" ")
    print()      


#        print(mat[i][j])
        
        
        
#        mat[i][j]=sum

#        print(mat[i][j])
            
#        sum=0
#        sum+=b
#        mat[i][j]=sum
#        print(mat[i][j],end=" ")
##        mat[i][j]=sum
#        print(mat[i][j])
                
                