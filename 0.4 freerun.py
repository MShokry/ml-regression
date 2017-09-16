#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 14:06:43 2017

@author: mshokry
"""

import pandas as pd
from sklearn import preprocessing 
from sklearn.linear_model import LinearRegression

data = pd.read_csv('sales.csv')
data.head()

A = np.array([[1,2,3],
               [4,5,6],
               [7,8,9]])
A[:,1]