# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 12:21:00 2021

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

# 3.2 plot variables:
plot_num_var(df=df,var=num_var_nonan)
get_violinplot_for_target(df=df,var=num_var_nonan,target='target')

# 3.3 enrich data:
#apply_value_log=create_log_var(df=df, num_var=num_var_nan + num_var_nonan)

import seaborn as sns
sns.displot(df.target)


# 3.4 corr
df_high_corr, high_corr_var=get_df_high_corr_target(df=df,target='target',min_cor=0.5)
get_scatter_for_target(df=df,var=high_corr_var,target='target')


### 4 pred data
# 4.1 
dependent=df['target']
independent=df.copy(deep=True)
independent.drop(columns='target',axis='columns', inplace=True)

# 4.2 split test train data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(independent,dependent,test_size=0.3, random_state=41, shuffle=True)

# 4.3 impute variables

# 4.3.4 final list
drop_list=[]
# drop_list.extend(drop_list_num)
# drop_list.extend(drop_list_str)
# drop_list.extend(drop_list_lowCor)
# drop_list.extend(drop_list_max_perc_rep)
# drop_list.extend(str_lst_scale)
# drop_list.extend(str_lst_flags) 

X_train.drop(columns=drop_list,axis='columns', inplace=True)


### 5 modeling
import multiprocessing
n_jobs = multiprocessing.cpu_count()-1

# 5.1 modeling on train
from sklearn.ensemble import RandomForestClassifier

# simple application
model = RandomForestClassifier(max_depth=12, n_estimators=400, random_state=41)
#model = RandomForestClassifier(max_depth=2, n_estimators=200, random_state=41)
#model = RandomForestClassifier(max_depth=20, n_estimators=1000, random_state=41)
model.fit(X_train,Y_train)


# Gridsearch
# from sklearn.model_selection import GridSearchCV
# params = {'n_estimators' : [200,400,600], 
#           'max_depth':[2,3,4], 
#           'random_state':[41]}

# model = RandomForestClassifier()
# model = GridSearchCV(model, params, cv = 5, n_jobs = n_jobs)
# model.fit(X_train,Y_train)

# print("Best params:{}".format(model.best_params_))
# model = model.best_estimator_
#Best params:{'max_depth': 2, 'n_estimators': 200, 'random_state': 41}


# RFECV
# model = RandomForestClassifier(n_estimators=200, max_depth=2, random_state=41)

# from sklearn.feature_selection import RFECV
# from sklearn.model_selection import StratifiedKFold

# min_features_to_select = 1  # Minimum number of features to consider

# rfecv = RFECV(estimator=model
#               ,step=min_features_to_select
#               ,cv=StratifiedKFold(10)
#               ,scoring='accuracy'
#               ,n_jobs = n_jobs)
# rfecv.fit(X_train, Y_train)
# print("Optimal number of features : %d" % rfecv.n_features_)
# print('Selected features: %s' % list(X_train.columns[rfecv.support_]))




#######################
from sklearn.metrics import accuracy_score, log_loss, auc, roc_curve, mean_squared_error, r2_score

Y_pred_train = model.predict(X_train)
Y_pred_proba_train = model.predict_proba(X_train)[:, 1]

Y_pred_test = model.predict(X_test)
Y_pred_proba_test = model.predict_proba(X_test)[:, 1]
[fpr, tpr, thr] = roc_curve(Y_test, Y_pred_proba_test)

idx = np.min(np.where(tpr > 0.95)) # index of the first threshold for which the sensibility > 0.95
import matplotlib.pyplot as plt
plt.figure()
plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot([0,fpr[idx]], [tpr[idx],tpr[idx]], 'k--', color='blue')
plt.plot([fpr[idx],fpr[idx]], [0,tpr[idx]], 'k--', color='blue')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
plt.ylabel('True Positive Rate (recall)', fontsize=14)
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()



# apply model to validation
df_val=pd.read_csv(full_path+"\\test.csv", index_col='id')


df_val['target']=model.predict(df_val)

#output
out=df_val['target']
out.to_csv(path_or_buf=full_path+"\\result.csv",index=True)


