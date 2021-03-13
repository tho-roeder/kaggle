# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 12:21:00 2021

@author: thoma
"""

"""
In the train and test data, features that belong to similar groupings are tagged as such in the 
feature names (e.g., ind, reg, car, calc). In addition, feature names include the postfix bin to 
indicate binary features and cat to indicate categorical features. Features without these 
designations are either continuous or ordinal. Values of -1 indicate that the feature was missing 
from the observation. The target columns signifies whether or not a claim was filed for that policy 
holder.

https://www.kaggle.com/johannesss/stacking-lightgbm-with-logisticregression
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
import seaborn as sns
sns.displot(df.target)

# 3.2 plot variables:
plot_num_var(df=df,var=num_var_nonan)
get_violinplot_for_target(df=df,var=num_var_nonan,target='target')

# 3.3 enrich data:
# for i in all_var:
#     df[i] = df[i].apply(lambda x: np.nan if x==-1 else x)

# for i in all_var:
#     df[i] = df[i].apply(lambda y: replace_value(y,-1))

#apply_value_log=create_log_var(df=df, num_var=num_var_nan + num_var_nonan)

# 3.4 corr
df_high_corr, high_corr_var=get_df_high_corr_target(df=df,target='target',min_cor=0.3)
get_scatter_for_target(df=df,var=high_corr_var,target='target')


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

# cat_features = [col for col in X_train.columns if '_cat' in col]
# for column in cat_features:
# 	temp = pd.get_dummies(pd.Series(X_train[column]))
# 	X_train = pd.concat([X_train,temp],axis=1)
# 	X_train = X_train.drop([column],axis=1)
#  # for column in cat_features:
# # 	temp = pd.get_dummies(pd.Series(X_test[column]))
# # 	X_test = pd.concat([X_test,temp],axis=1)
# # 	X_test = X_test.drop([column],axis=1)


# X_train = pd.DataFrame(X_train)
# X_test = pd.DataFrame(X_test)
# print('Training data: ', X_train.values.shape, ',  Test data: ', X_test.values.shape)

    
cat_features = [col for col in X_train.columns if '_cat' in col]
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
column_trans=make_column_transformer(
    (OneHotEncoder(sparse=False),cat_features),
    remainder='passthrough'    
    )
X_train=pd.DataFrame(column_trans.fit_transform(X_train))
#X_train_new = pd.concat([X_train,X_train_ohe],axis=1, ignore_index=True)

X_test=pd.DataFrame(column_trans.transform(X_test))

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
from sklearn.model_selection import GridSearchCV

# Gridsearch
# params = {'n_estimators' : [200,400,600], 
#           'max_depth':[2,3,4], 
#           'random_state':[41]}
#Best params:{'max_depth': 2, 'n_estimators': 200, 'random_state': 41}

params = {'n_estimators' : [200], 
          'max_depth':[2], 
          'random_state':[41]}


model = RandomForestClassifier()
model = GridSearchCV(model, params, cv = 5, n_jobs = n_jobs)
model.fit(X_train,Y_train)

print("Best params:{}".format(model.best_params_))
model = model.best_estimator_


# model = RandomForestClassifier(max_depth=2, n_estimators=200, random_state=41)

# RFECV
# runs for a very long time

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


##################
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

min_features_to_select = 1  # Minimum number of features to consider

rfecv = RFECV(estimator=model
              ,step=min_features_to_select
              ,cv=StratifiedKFold(5)
              ,scoring='accuracy')
rfecv.fit(X_train, Y_train)

print("Optimal number of features : %d" % rfecv.n_features_)
print('Selected features: %s' % list(X_train.columns[rfecv.support_]))

# ['ps_car_13']

# model_param= dict(zip(np.array(X_train.columns),rfecv.coef_[0]))
# model_param['intercept']=rfecv.intercept_[0]


##############
from sklearn.model_selection import StratifiedShuffleSplit

# Training
n_splits = 3
splitter = StratifiedShuffleSplit(n_splits=n_splits)

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
#####################
lgb_params2 = {}
lgb_params2['seed'] = 1974
lgb_params2['learning_rate'] = 0.02
lgb_params2['n_estimators'] = 1500
lgb_params2['colsample_bytree'] = 0.3   
lgb_params2['subsample'] = 0.7
lgb_params2['subsample_freq'] = 2

lgb_params2['num_leaves'] = 16
#####################
lgb_params3 = {}
lgb_params3['seed'] = 1974
lgb_params3['learning_rate'] = 0.02
lgb_params3['n_estimators'] = 1500
lgb_params3['max_depth'] = 4

model1 = lgb.LGBMClassifier(**lgb_params1)
model2 = lgb.LGBMClassifier(**lgb_params2)
model3 = lgb.LGBMClassifier(**lgb_params3)

scores = []

for i, (fit_index, val_index) in enumerate(splitter.split(X_train, Y_train)):
    X_fit = X_train.iloc[fit_index,:].copy()
    y_fit = Y_train.iloc[fit_index].copy()
    X_val = X_train.iloc[val_index,:].copy()
    y_val = Y_train.iloc[val_index].copy()

    model1.fit(
        X_fit,
        y_fit,
        eval_set=[(X_val, y_val)],
        eval_metric=gini_lgb,
        early_stopping_rounds=50,
        verbose=False  )
    # model1 = lgb.train(params1, 
    #               train_set       = lgb.Dataset(X_fit, label=y_fit), 
    #               num_boost_round = 200,
    #               valid_sets      = lgb.Dataset(X_val, label=y_val),
    #               verbose_eval    = 50, 
    #               feval           = gini_lgb,
    #               early_stopping_rounds = 50)
    #y_val_predprob1 = model1.predict(X_val, num_iteration=model1.best_iteration)
    y_val_predprob1 = model1.predict_proba(X_val)[:,1]
    score = gini_normalized(y_val, y_val_predprob1)
    scores.append(score)
    print('Fold {} model 1: {} gini'.format(i+1, score))
    #x_test_pred1   = model1.predict(X_test, num_iteration=model1.best_iteration)
    x_test_pred1   = model1.predict_proba(X_test)[:,1] 
    #x_train_pred1  = model1.predict(X_train, num_iteration=model1.best_iteration)
    x_train_pred1  = model1.predict_proba(X_train)[:,1] 
    # X_logreg_test[:, i * num_models]  = x_test_pred1
    # X_logreg_train[:, i * num_models] = x_train_pred1
    
    model2.fit(
        X_fit,
        y_fit,
        eval_set=[(X_val, y_val)],
        eval_metric=gini_lgb,
        early_stopping_rounds=50,
        verbose=False  )
    y_val_predprob2 = model2.predict_proba(X_val)[:,1]
    score = gini_normalized(y_val, y_val_predprob2)
    scores.append(score)
    print('Fold {} model 2: {} gini'.format(i+1, score))
    x_test_pred2   = model2.predict_proba(X_test)[:,1] 
    x_train_pred2  = model2.predict_proba(X_train)[:,1] 
    # X_logreg_test[:, i * num_models + 1]  = x_test_pred2
    # X_logreg_train[:, i * num_models + 1] = x_train_pred2
    
    model3.fit(
        X_fit,
        y_fit,
        eval_set=[(X_val, y_val)],
        eval_metric=gini_lgb,
        early_stopping_rounds=50,
        verbose=False  )
    y_val_predprob3 = model3.predict_proba(X_val)[:,1]
    score = gini_normalized(y_val, y_val_predprob3)
    scores.append(score)
    print('Fold {} model 3: {} gini'.format(i+1, score))
    x_test_pred3   = model3.predict_proba(X_test)[:,1] 
    x_train_pred3  = model3.predict_proba(X_train)[:,1] 
    # X_logreg_test[:, i * num_models + 2]  = x_test_pred3
    # X_logreg_train[:, i * num_models + 2] = x_train_pred3



params1 = {'learning_rate': 0.09, 'max_depth': 6, 'boosting': 'gbdt', 'objective': 'binary', 
          'metric': 'auc', 'num_leaves': 40, 'min_data_in_leaf': 200, 'max_bin': 100, 'colsample_bytree' : 0.5,   
          'subsample': 0.7, 'subsample_freq': 2, 'verbose':-1, 'is_training_metric': False, 'seed': 1974}
params2 = {'learning_rate': 0.12, 'max_depth': 4, 'verbose':-1, 'num_leaves':16,
           'is_training_metric': False, 'seed': 1974} 
params3 = {'learning_rate': 0.11, 'subsample': 0.8, 'boosting': 'gbdt', 'objective': 'binary', 
          'metric': 'auc', 'subsample_freq': 10, 'colsample_bytree': 0.6, 'max_bin': 10, 
           'min_child_samples': 500,'verbose':-1, 'is_training_metric': False, 'seed': 1974}  

num_models = 3
log_model       = LogisticRegression()
X_logreg_train  = np.zeros((X_train.shape[0], n_splits * num_models))
X_logreg_test   = np.zeros((X_test.shape[0], n_splits * num_models))









## check performance:

from sklearn.metrics import accuracy_score, log_loss, auc, roc_curve, mean_squared_error, r2_score

Y_pred_train = rfecv.predict(X_train)
Y_pred_proba_train = rfecv.predict_proba(X_train)[:, 1]

Y_pred_test = rfecv.predict(X_test)
Y_pred_proba_test = rfecv.predict_proba(X_test)[:, 1]
[fpr, tpr, thr] = roc_curve(Y_test, Y_pred_proba_test)

print('Train/Test split results:')
print(rfecv.__class__.__name__+" Train mean squared error is %.2f" % mean_squared_error(Y_train, Y_pred_train))
print(rfecv.__class__.__name__+" Train R2 is %.2f" % r2_score(Y_train, Y_pred_train))
print(rfecv.__class__.__name__+" Test mean squared error is %.2f" % mean_squared_error(Y_test, Y_pred_test))
print(rfecv.__class__.__name__+" Test R2 is %.2f" % r2_score(Y_test, Y_pred_test))
print(rfecv.__class__.__name__+" Test accuracy is %2.3f" % accuracy_score(Y_test, Y_pred_test))
print(rfecv.__class__.__name__+" Test log loss is %2.3f" % log_loss(Y_test, Y_pred_proba_test))
print(rfecv.__class__.__name__+" Test auc is %2.3f" % auc(fpr, tpr))



idx = np.min(np.where(tpr > 0.95)) # index of the first threshold for which the sensibility > 0.95

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

print("Using a threshold of %.3f " % thr[idx] + "guarantees a sensitivity of %.3f " % tpr[idx] +  
      "and a specificity of %.3f" % (1-fpr[idx]) + 
      ", i.e. a false positive rate of %.2f%%." % (np.array(fpr[idx])*100))



# apply model to validation
df_val=pd.read_csv(full_path+"\\test.csv", index_col='id')

df_val['target']=model.predict(df_val)

#output

# submission = pd.DataFrame({
#     'id': test_ids,
#     'target': 0
# })


out=df_val['target']
out.to_csv(path_or_buf=full_path+"\\result.csv",index=True)



####################

from numba import jit
@jit
def gini(y_true, y_prob):
    """
    Original author CPMP : https://www.kaggle.com/cpmpml
    In kernel : https://www.kaggle.com/cpmpml/extremely-fast-gini-computation
    """
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini    

def gini_normalized(a, p):
     return gini(a, p) / gini(a, a)
    
def gini_lgb(preds, dtrain):
    y = dtrain
    score = gini(y, preds) / gini(y, y)
    return 'gini', score, True    
    
def gini_lgb_train(preds, dtrain):
    y = list(dtrain.get_label())
    score = gini(y, preds) / gini(y, y)
    return 'gini', score, True     

