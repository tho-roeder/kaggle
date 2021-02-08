# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 21:12:35 2021

@author: thoma
"""

path="\\Desktop\\VM share\\Python\\files\\Kaggle\\safe driver"

import pandas as pd
import os

# source data
df=pd.read_csv(os.getcwd()+path+"\\train.csv", index_col='id')

independent=df.loc[:,'ps_ind_01':'ps_calc_20_bin']
dependent=df.loc[:,'target']



from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(independent, dependent, test_size = 0.3, random_state = 41)


import lightgbm as lgb


d_train = lgb.Dataset(X_train, label=Y_train)
params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 50
params['max_depth'] = 10

model = lgb.train(params, d_train, 100)

y_pred=model.predict(X_test)


