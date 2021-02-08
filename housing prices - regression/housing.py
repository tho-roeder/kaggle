# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
import os

# source data
df=pd.read_csv(os.getcwd()+"\\Desktop\\VM share\\Python\\Kaggle\\housing prices - regression"+"\\train.csv", index_col='Id')

# select variables:    
all_columns=df.columns
independent=df[all_columns[:-1]]
dependent=df[all_columns[-1:]]

# check input data
detect_na=df.isna().any()
issue=detect_na[detect_na == True].index
issue_cnt=df.isnull().sum()

for i in issue:
    print(df[i].value_counts())

print(df.info)
print(type(df))
print(df.dtypes)

#type dependent imputation
# independent_miss=independent.fillna('missing')
# independent_dum=pd.get_dummies(independent_miss)
# independent_treated=independent_dum



# split test train data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(independent_treated,dependent,test_size=0.3, random_state=41, shuffle=True)

# modeling
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

# assessing result
reg.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(Y_test, reg.predict(X_test)))

