# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
import os


df=pd.read_csv(os.getcwd()+"\\Desktop\\VM share\\Python\\Kaggle\\housing prices - regression"+"\\train.csv")


detect_na=df.isna().any()

issue=detect_na[detect_na == True].index

for i in issue:
    print(df[i].value_counts())
    
    
all_columns=df.columns()


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split()
