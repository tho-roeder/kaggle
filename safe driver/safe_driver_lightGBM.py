# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 13:05:24 2021

@author: thoma
"""


### 1. import helper
import os
path="C:\\Users\\thoma\\Desktop\\VM share\\Python\\Kaggle\\helper"
os.chdir(path)
%run helper.py


### 2. source data
full_path="C:\\Users\\thoma\\Desktop\\VM share\\Python\\files\\Kaggle\\safe driver"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv(full_path+"\\train.csv", index_col='id')


### 3. exploratory data analysis
# 3.1 create overview of data set:
# getGeneralInformation(df)
all_var, num_var_nan, num_var_nonan, str_var_nan, str_var_nonan= pre_work(df)   
value_counts=get_var_value_counts(df=df,var=all_var)
# df_sample=df.iloc[1:500,:]
# Target
# import seaborn as sns
# sns.displot(df.target)

# 3.2 plot variables:
# plot_num_var(df=df,var=num_var_nonan)
# get_violinplot_for_target(df=df,var=num_var_nonan,target='target')

# 3.3 enrich data:
# for i in all_var:
#     df[i] = df[i].apply(lambda x: np.nan if x==-1 else x)

# for i in all_var:
#     df[i] = df[i].apply(lambda y: replace_value(y,-1))

#apply_value_log=create_log_var(df=df, num_var=num_var_nan + num_var_nonan)

# 3.4 corr
df_high_corr, high_corr_var=get_df_high_corr_target(df=df,target='target',min_cor=0.3)
# get_scatter_for_target(df=df,var=high_corr_var,target='target')


### 4 pred data
# 4.1 
dependent=df['target']
independent=df.copy(deep=True)
independent.drop(columns='target',axis='columns', inplace=True)

# 4.2 split test train data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(independent,dependent,test_size=0.3, random_state=41, shuffle=True)

# 4.3 impute/transform variables

col_to_drop = X_train.columns[X_train.columns.str.startswith('ps_calc_')]
X_train     = X_train.drop(col_to_drop, axis=1)  
#X_test      = X_test.drop(col_to_drop, axis=1)  
    
cat_features = [col for col in X_train.columns if '_cat' in col]
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
column_trans=make_column_transformer(
    (OneHotEncoder(),cat_features),
    remainder='passthrough'    
    )
X_train=column_trans.fit_transform(X_train)
X_test=column_trans.transform(X_test)

# X_train=pd.DataFrame(column_trans.fit_transform(X_train))
# X_test=pd.DataFrame(column_trans.transform(X_test))


import lightgbm as lgb
lgb_params1 = {}
lgb_params1['seed'] = 1974
lgb_params1['learning_rate'] = 0.02
lgb_params1['n_estimators'] = 300
lgb_params1['colsample_bytree'] = 0.7   
lgb_params1['subsample'] = 0.7
lgb_params1['subsample_freq'] = 12

lgb_params1['max_bin'] = 10
lgb_params1['min_child_samples'] = 600

lgb_params1['is_unbalance'] = True

# params = {}
# params['learning_rate'] = 0.003
# params['boosting_type'] = 'gbdt'
# params['objective'] = 'binary'
# params['metric'] = 'binary_logloss'
# params['sub_feature'] = 0.5
# params['num_leaves'] = 10
# params['min_data'] = 50
# params['max_depth'] = 10


model = lgb.LGBMClassifier(**lgb_params1)

model.fit(X_train, Y_train, verbose=True)

model.score(X_test,Y_test)


# apply model to validation
df_val=pd.read_csv(full_path+"\\test.csv", index_col='id')
new2=df_val.copy(deep=True)
df_val = df_val.drop(col_to_drop, axis=1)  
df_val=column_trans.transform(df_val)

new2[['predict_0','predict_1']]=model.predict_proba(df_val)

# df_val['target'].sum()
new2.rename(columns={'predict_1':'target'}, inplace=True)

out=new2['target']
out.to_csv(path_or_buf=full_path+"\\result.csv",index_label='id',index=True)

#0.27948
