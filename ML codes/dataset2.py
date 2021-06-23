# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 16:28:10 2019

@author: Mahnoor Ali
"""

import pandas as pd
from sklearn import linear_model
#from sklearn import tree
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'C:\Users\Mahnoor Ali\Downloads\Admission_Predict.csv')
#1-) training using linear model module
label = df['Chance of Admit']
features = df.drop('Serial No.','Chance of Admit',axis=2)
regr = linear_model.LinearRegression()
regr.fit(features, label)
print(regr.predict([[337,118,4,4.5,4.5,9.65,1]]))

