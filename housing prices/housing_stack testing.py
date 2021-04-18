# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 11:39:23 2021

@author: thoma
"""


# use rfe
# use pipeline
# use ensemble
# allow na
# build function to add basis impact while grid searching (or importance of Hyperparameter)
# from sklearn.metrics import roc_auc_score
# use k fold instead of test/train split


### 1. import helper
import os
path="C:\\Users\\thoma\\Desktop\\VM share\\Python\\Kaggle\\helper"
os.chdir(path)
%run helper.py


### 2. source data
full_path="C:\\Users\\thoma\\Desktop\\VM share\\Python\\files\\Kaggle\\housing prices"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv(full_path+"\\train.csv", index_col='Id')


### 3. exploratory data analysis
# 3.1 create overview of data set:
#getGeneralInformation(df)
all_var, num_var_nan, num_var_nonan, str_var_nan, str_var_nonan= pre_work(df)   

# 3.2 plot variables:
# plot_str_var(df=df,var=str_var_nan)
# plot_str_var(df=df,var=['Street'])
# plot_num_var(df=df,var=num_var_nan)

# 3.3 enrich data:
# Variable treement String:  
str_lst_flags=['Alley','Street','MiscFeature','FireplaceQu','GarageQual','Fence']
for i in str_lst_flags:
    df['f_'+str(i)]=df[i].apply(lambda x: create_flags(x))

# feture engineering a new feature "TotalSF"
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
df['Total_Bathrooms'] = (df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))

# 3.4 corr
df_high_corr, high_corr_var=get_df_high_corr_target(df=df,target='SalePrice',min_cor=0.5)
# get_scatter_for_target(df=df,var=high_corr_var,target='SalePrice')

# 3.5 impute without data leakage
drop_list_str,impute_value_str=impute_var_v4(df=df,var=str_var_nan,perc_drop=1,style='value',value='NA')

# 3.6 check for missing
#num_var_nonan.remove('SalePrice')
print(df[str_var_nan + str_var_nonan].isna().sum())
print(df[num_var_nan + num_var_nonan].isna().sum())


### 4 pred data
# 4.1 split dependent and independent
dependent=df['SalePrice']
independent=df.copy(deep=True)
independent.drop(columns='SalePrice',axis='columns', inplace=True)

dependent_Log1p = np.log1p(dependent)

# 4.2 
# 4.2.1 build pipeline
num_var_nonan.remove('SalePrice')

nominal=['MSSubClass','MSZoning','Street', 'Alley','LandContour','Utilities','LotConfig','Neighborhood',
         'Condition1','Condition2',
         'BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation',
         'Heating','CentralAir','Electrical','GarageType','PavedDrive','Fence','MiscFeature','SaleType',
         'SaleCondition',
         'f_Alley','f_Street','f_MiscFeature','f_FireplaceQu','f_GarageQual','f_Fence']
categorial=['LotShape','LandSlope','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
            'BsmtFinType2','HeatingQC','KitchenQual','Functional','FireplaceQu','GarageFinish',
            'GarageQual','GarageCond','PoolQC']
category_array=[
    ['Reg','IR1', 'IR2', 'IR3'],
    ['Gtl','Mod','Sev'],
    ['Ex','Gd','TA','Fa','Po'],
    ['Ex','Gd','TA','Fa','Po'],
    ['Ex','Gd','TA','Fa','Po','NA'],
    ['Ex','Gd','TA','Fa','Po','NA'],
    ['Gd', 'Av', 'Mn', 'No','NA'],
    ['GLQ', 'ALQ', 'BLQ','Rec', 'LwQ', 'Unf', 'NA'],
    ['GLQ', 'ALQ', 'BLQ','Rec', 'LwQ', 'Unf', 'NA'],
    ['Ex','Gd','TA','Fa','Po'],
    ['Ex','Gd','TA','Fa','Po'],
    ['Typ','Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal'],
    ['Ex','Gd','TA','Fa','Po','NA'],
    ['Fin','RFn','Unf','NA'],
    ['Ex','Gd','TA','Fa','Po','NA'],
    ['Ex','Gd','TA','Fa','Po','NA'],
    ['Ex','Gd','TA','Fa','NA']]

apply_value_log=create_log_var(df=df, num_var=num_var_nan + num_var_nonan)

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import FunctionTransformer

column_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), nominal),
    (OrdinalEncoder(categories=category_array), categorial),
    (SimpleImputer(strategy='median'), num_var_nan),
    (SimpleImputer(strategy='median'), num_var_nonan),
   #(FunctionTransformer(np.log1p, validate=False),apply_value_log),
    remainder='passthrough',
    verbose=True)
independent_trans=column_trans.fit_transform(independent)

# from pandas import DataFrame
# missing_vars=DataFrame(independent_trans).isna().sum()

# 4.2.2 best model
# do this on train vs. test to check parameters
best_model=pre_eval_models(
    type_model='regression',
    dependent=independent_trans, 
    inpendent=dependent,
    scoring='neg_root_mean_squared_error',
    cv=10)

best_model_log=pre_eval_models(
    type_model='regression',
    dependent=independent_trans, 
    inpendent=dependent_Log1p,
    scoring='neg_root_mean_squared_error',
    cv=10)

# 4.2.3 split test train data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(independent,dependent,test_size=0.3, random_state=41, shuffle=True)

X_train_trans=column_trans.fit_transform(X_train)
X_test_trans=column_trans.transform(X_test)

Y_train_Log1p = np.log1p(Y_train)
Y_test_Log1p = np.log1p(Y_test)


# 4.3 Best Parameters
# 4.3.0 determine final optimizaion metric
# Root-Mean-Squared-Error (RMSE) 
# https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

#model = XGBRegressor(n_estimators=75)
model = XGBRegressor(n_estimators=100)

model.fit(X_train_trans,Y_train)
model.score(X_train_trans, Y_train)
model.score(X_test_trans,Y_test)

# 4.3.1 GridSearchCV
from sklearn.model_selection import GridSearchCV

from xgboost import XGBRegressor
params={
        'objective': ['reg:gamma'] #['reg:gamma', 'reg:squarederror', 'reg:squaredlogerror']
        ,'learning_rate': [0.25] #[0.0001, 0.001, 0.01, 0.1, 0.15, 0.2, 0.25, 0.3]
        ,'n_estimators': [100] #[50,100,200,500,1000,2000,5000]
        ,'max_depth': [11] #[3,7,11,12,13]
        ,'booster': ['dart'] #['gbtree','gblinear','dart']
        # ,'min_child_weight': [0.5] #[0.5,1,1.5,2.5,5]
        # ,'gamma': [0] #[0,1,5,10]
        # ,'colsample_bytree' : [1] # [0.2,0.4,0.8,1.0]
        # ,'subsample':[1] #[0.2,0.4,0.8,1]
        # ,'reg_alpha': [0] #[0.0,0.1,0.2,0.5,0.7,0.9,1.0]
        # ,'reg_lambda': [1] # [0,0.1,0.2,0.5,0.7,0.9, 1]
        }

model = GridSearchCV(XGBRegressor(), param_grid=params, scoring='neg_root_mean_squared_error', cv = 5, n_jobs =2)
model.fit(X_train_trans, Y_train)
#model.best_params_
#model.cv_results_

for param, score in zip(model.cv_results_['params'], model.cv_results_['mean_test_score']):
    print (param, score)
print(model.best_params_)

#{'booster': 'dart', 'learning_rate': 0.25, 'max_depth': 11, 'n_estimators': 100, 'objective': 'reg:gamma'} -26575.280686733768
    # Mean squared error Train: 71111.62
    # Coefficient of determination R2 Train: 1.00
    # Mean squared error Test: 573296440.03
    # Coefficient of determination R2 Test: 0.89


# 4.3.2 RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
params={
        'objective': ['reg:gamma', 'reg:squarederror', 'reg:squaredlogerror']
        ,'booster': ['gbtree','gblinear','dart']
        ,'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.15, 0.2, 0.25, 0.3]
        ,'n_estimators': [50,100,200,500,1000,2000,5000]
        ,'max_depth': [3,7,11,12,13,15,30]
        }

model =RandomizedSearchCV(XGBRegressor(), params, scoring='neg_root_mean_squared_error',cv = 5, n_jobs =2)
model.fit(X_train_trans, Y_train)

for param, score in zip(model.cv_results_['params'], model.cv_results_['mean_test_score']):
    print (param, score)
print(model.best_params_)
#1: {'objective': 'reg:squarederror', 'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.2, 'booster': 'dart'}
#2: {'objective': 'reg:gamma', 'n_estimators': 1000, 'max_depth': 15, 'learning_rate': 0.2, 'booster': 'dart'}
#3 with scoring set: {'objective': 'reg:gamma', 'n_estimators': 2000, 'max_depth': 7, 'learning_rate': 0.1, 'booster': 'gbtree'}
#4: {'objective': 'reg:gamma', 'n_estimators': 2000, 'max_depth': 7, 'learning_rate': 0.15, 'booster': 'dart'} -26575.447444184923
    # Mean squared error Train: 42899.02
    # Coefficient of determination R2 Train: 1.00
    # Mean squared error Test: 509452737.27
    # Coefficient of determination R2 Test: 0.90


# 4.3.3 Optuna
def objective(trial):
    from sklearn.model_selection import cross_val_score
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_squared_error
    _learning_rate = trial.suggest_float("learning_rate",0.0001,0.3)
    _n_estimators = trial.suggest_int("n_estimators",50,5000)
    _max_depth= trial.suggest_int("max_depth", 3,30)
    # print(_learning_rate)
    # print(_n_estimators)
    # print(_max_depth)
    model = XGBRegressor(objective='reg:gamma'
                          ,booster = 'dart'
                          ,learning_rate=_learning_rate
                          ,max_depth=_max_depth
                          ,n_estimators=_n_estimators
                          )
    score= cross_val_score(
        model, X_train_trans, Y_train, cv=5, scoring="neg_root_mean_squared_error"
    ).mean()
    return score

optuna.logging.set_verbosity(0)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)
params = study.best_params
best_score = study.best_value
print(f"Best score:{best_score} \nOptimized parameters: {params}")


# _learning_rate = trial.suggest_float("learning_rate",0.1,0.5)
# _n_estimators = trial.suggest_int("n_estimators",1000,3000)
# _max_depth= trial.suggest_int("max_depth", 5,30)

# Best score:-27056.959313538195 
# Optimized parameters: {'learning_rate': 0.31590936013482734, 'n_estimators': 1651, 'max_depth': 20}

# Mean squared error Train: 11141.13
# Coefficient of determination R2 Train: 1.00
# Mean squared error Test: 601828629.39
# Coefficient of determination R2 Test: 0.88

# Best score:-26122.913740461874 
# Optimized parameters: {'learning_rate': 0.2710320608415172, 'n_estimators': 2081, 'max_depth': 5}

# Mean squared error Train: 44621.29
# Coefficient of determination R2 Train: 1.00
# Mean squared error Test: 566964778.93
# Coefficient of determination R2 Test: 0.89


model = XGBRegressor(objective='reg:gamma'
                      ,booster = 'dart'
                      ,learning_rate=0.2710320608415172
                      ,max_depth=5
                      ,n_estimators=2081
                      )
model.fit(X_train_trans, Y_train)


# 4.4 Check metrics
from sklearn.metrics import mean_squared_error, r2_score
  
#Train
print('Mean squared error Train: %.2f'
      % mean_squared_error(Y_train, model.predict(X_train_trans)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination R2 Train: %.2f'
      % r2_score(Y_train, model.predict(X_train_trans)))

#Test
print('Mean squared error Test: %.2f'
      % mean_squared_error(Y_test, model.predict(X_test_trans)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination R2 Test: %.2f'
      % r2_score(Y_test, model.predict(X_test_trans)))



# 4.5 various models:
# 4.5.1 prepare stack:

from sklearn.model_selection import KFold
cv = KFold(n_splits=4, random_state=random_state)

# svr = SVR(**svr_params)
# ridge = Ridge(**ridge_params, random_state=random_state)
# lasso = Lasso(**lasso_params, random_state=random_state)
# lgbm = LGBMRegressor(**lgbm_params, random_state=random_state)
# rf = RandomForestRegressor(**rf_params, random_state=random_state)

random_state=41;
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

svr = SVR()
ridge = Ridge(random_state=random_state)
lasso = Lasso(random_state=random_state)
lgbm = LGBMRegressor(random_state=random_state)
rf = RandomForestRegressor(random_state=random_state)
XGB = XGBRegressor(objective='reg:gamma'
                      ,booster = 'dart'
                      ,learning_rate=0.2710320608415172
                      ,max_depth=5
                      ,n_estimators=2081
                      )

from mlxtend.regressor import StackingCVRegressor
from sklearn.linear_model import LinearRegression

model = StackingCVRegressor(
    regressors=[lgbm, rf, XGB],
    meta_regressor=XGB,
    random_state=random_state,
    cv=cv,
    n_jobs=-1,
)

model.fit(X_train_trans,Y_train)


# svr_scores = cross_val_score(
#     svr, X_train, y_train, cv=cv, n_jobs=-1, error_score="neg_root_mean_squared_error"
# )

# 4.6 compare all models

# scores = [svr_scores, ridge_scores, lasso_scores, lgbm_scores, rf_scores, stack_scores]
# models = ["SVR", "RIDGE", "LASSO", "LGBM", "RF", "STACK"]
# score_medians = [
#     round(np.median([mean for mean in modelscore]), 5) for modelscore in scores
# ]

# fig, ax = plt.subplots(figsize=(14, 8))

# vertical_offset = 0.001

# ax.set_title("Model Score Comparison")
# bp = sns.boxplot(x=models, y=scores, ax=ax)


# for xtick in bp.get_xticks():
#     bp.text(
#         xtick,
#         score_medians[xtick] - vertical_offset,
#         score_medians[xtick],
#         horizontalalignment="center",
#         size=18,
#         color="w",
#         weight="semibold",
#     )

# plt.show()






########


# 5. Apply model
# apply model to validation
df_val=pd.read_csv(full_path+"\\test.csv", index_col='Id')

str_lst_flags=['Alley','Street','MiscFeature','FireplaceQu','GarageQual','Fence']
for i in str_lst_flags:
    df_val['f_'+str(i)]=df_val[i].apply(lambda x: create_flags(x))

# feture engineering a new feature "TotalSF"
df_val['TotalSF'] = df_val['TotalBsmtSF'] + df_val['1stFlrSF'] + df_val['2ndFlrSF']
df_val['Total_Bathrooms'] = (df_val['FullBath'] + (0.5 * df_val['HalfBath']) + df_val['BsmtFullBath'] + (0.5 * df_val['BsmtHalfBath']))

for i in impute_value_str.keys():
    df_val[i].fillna(impute_value_str[i], inplace=True)

impute_var_v4(df=df_val,var=str_var_nonan,perc_drop=1,style='mode')
#impute_var_v4(df=df_val,var=num_var_nonan,perc_drop=1,style='median')

new=df_val.isna().sum().sort_values()

independent_trans_val=column_trans.transform(df_val)
import numpy
independent_trans_val=numpy.nan_to_num(independent_trans_val, copy=True, nan=0.0)

# from pandas import DataFrame
# print(DataFrame(independent_trans_val).isna().sum())


#df_val['predicted']=np.expm1(model.predict(independent_trans_val))
df_val['SalePrice']=model.predict(independent_trans_val)

#output
out=df_val['SalePrice']
out.to_csv(path_or_buf=full_path+"\\result.csv",index=True)



