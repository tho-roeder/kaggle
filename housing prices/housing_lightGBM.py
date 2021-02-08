# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 22:04:56 2021

@author: thoma
"""

path="\\Desktop\\VM share\\Python\\files\\Kaggle\\housing prices"

import pandas as pd
import os

# source data
df=pd.read_csv(os.getcwd()+path+"\\train.csv", index_col='Id')


# select variables:    
all_columns=df.columns
independent=df[all_columns[:-1]]
dependent=df[all_columns[-1:]]

train_independent_dum=pd.get_dummies(independent)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(train_independent_dum, dependent, test_size = 0.3, random_state = 41)


import lightgbm as lgb


d_train = lgb.Dataset(X_train, label=Y_train)
params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'rmse'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 50
params['max_depth'] = 10

model = lgb.train(params, d_train, 100)

X_test['predict']=model.predict(X_test)
final_df=X_test.merge(Y_test, on='Id')
final_df['diff']=final_df['predict']-final_df['SalePrice']

import matplotlib.pyplot as plt
plt.scatter(final_df['predict'],final_df['SalePrice'])
plt.show()


#output
test_df=pd.read_csv(os.getcwd()+path+"\\test.csv", index_col='Id')
test_independent_dum=pd.get_dummies(test_df)

test_df['SalePrice']= model.predict(test_independent_dum, predict_disable_shape_check=True)
out=test_df['SalePrice']
out.to_csv(path_or_buf=os.getcwd()+path+"\\result.csv",index=True)


