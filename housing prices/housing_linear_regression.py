# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# tod do: treatment of missing values

path="\\Desktop\\VM share\\Python\\files\\Kaggle\\housing prices"

import pandas as pd
import os

# source data
df=pd.read_csv(os.getcwd()+path+"\\train.csv", index_col='Id')

# select variables:    
all_columns=df.columns
independent=df[all_columns[:-1]]
dependent=df[all_columns[-1:]]

# check input data
detect_na=independent.isna().any()
issue=detect_na[detect_na == True].index
issue_cnt=independent.isnull().sum()
issue=issue_cnt[issue_cnt >= 0].index
no_issue=issue_cnt[issue_cnt == 0].index

for i in issue:
    print(df[i].value_counts())

print(df.info)
print(type(df))
print(df.dtypes)


#type dependent imputation
independent_input=independent[no_issue]
# independent_miss=independent.fillna('missing')
independent_dum=pd.get_dummies(independent_input)

independent_treated=independent_dum

# split test train data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(independent_treated,dependent,test_size=0.3, random_state=41, shuffle=True)

# modeling
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

# assessing result
X_test['predict']=reg.predict(X_test)
final_df=X_test.merge(Y_test, on='Id')
final_df['diff']=final_df['predict']-final_df['SalePrice']
#from sklearn.metrics import classification_report
#print(classification_report(Y_test, reg.predict(X_test)))

import matplotlib.pyplot as plt
plt.scatter(final_df['predict'],final_df['SalePrice'])
plt.show()


#output
df=pd.read_csv(os.getcwd()+path+"\\test.csv")
test_independent_input=df[no_issue]
test_issue_cnt=test_independent_input.isnull().sum()
# independent_miss=independent.fillna('missing')
test_independent_dum=pd.get_dummies(test_independent_input)

df['SalePrice']= reg.predict(test_independent_dum)
out=df[['Id', 'SalePrice']]
out.to_csv(path_or_buf=os.getcwd()+path+"\\result.csv",index=False)
