# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 11:39:23 2021

@author: thoma
"""
#other way to optimize: https://www.kaggle.com/mustafacicek/lightgbm-xgboost-parameter-tuning-bayessearchcv
#VarianceThreshold
#PowerTransformer

### 1. import helper
import os
path="C:\\Users\\thoma\\Desktop\\VM share\\Python\\Kaggle\\helper"
os.chdir(path)
%run helper.py


### 2. source data
full_path="C:\\Users\\thoma\\Desktop\\VM share\\Python\\files\\Kaggle\\housing prices"
import pandas as pd
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

import numpy as np
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
from xgboost import XGBRegressor
#model = XGBRegressor(n_estimators=75)
model = XGBRegressor(n_estimators=100)

model.fit(X_train_trans,Y_train)
model.score(X_train_trans, Y_train)
model.score(X_test_trans,Y_test)

bm = sel_reg_model_features_v2(model,X_train_trans,Y_train,X_test_trans,Y_test,step=2,min_features_to_select=12) #0.8784482325344447 step:  2 min_features_to_select:  12
X_train_trans_reduce=bm.transform(X_train_trans)


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
#model.fit(X_train_trans_reduce, Y_train)


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
#5: {'objective': 'reg:gamma', 'n_estimators': 100, 'max_depth': 12, 'learning_rate': 0.25, 'booster': 'gbtree'} -26788.80422612557


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

# 4.3.4 BayesSearchCV
#https://www.kaggle.com/mustafacicek/lightgbm-xgboost-parameter-tuning-bayessearchcv
# from skopt import BayesSearchCV
# from skopt.space import Real, Integer
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from datetime import datetime

# gbr = GradientBoostingRegressor(n_estimators = 200, learning_rate = 0.1, loss = "huber", 
#                                 random_state = 42, n_iter_no_change = 20)

# kf = KFold(n_splits = 10, shuffle = True, random_state = 42)

# def rmse_cv(model, X = X_train_trans, y = Y_train, cv=kf):    
#     return np.sqrt(-cross_val_score(model, X, y, scoring = 'neg_mean_squared_error', cv = cv)).mean()


# rmse_cv(gbr)

# def on_step(optim_result):
#     """
#     Callback meant to view scores after
#     each iteration while performing Bayesian
#     Optimization in Skopt"""
#     score = opt.best_score_
#     print("best score: %s" % score)
#     if score >= -0.11:
#         print('Interrupting!')
#         return True


%%time
# start = datetime.now()
# print(start)

# opt = BayesSearchCV(gbr,
#                     {
#                         "max_depth": Integer(3, 13),
#                         "max_features": Real(0.1, 1, prior = "log-uniform"),
#                         "subsample": Real(0.25, 1),
#                         "min_samples_split": Integer(20, 120),
#                         "min_samples_leaf": Integer(1, 10),
#                         "alpha": Real(0.75, 0.95),
#                         "min_impurity_decrease": Real(0, 0.5)
#                     },
#                     n_iter = 150,
#                     cv = kf,
#                     n_jobs = -1,
#                     scoring = "neg_root_mean_squared_error",
#                     random_state = 42
#                     )


# opt.fit(X_train_trans, Y_train, callback = on_step)


# end = datetime.now()
# print(end)

# print("Best Score is: ", opt.best_score_, "\n")

# print("Best Parameters: ", opt.best_params_, "\n")


#Best Parameters:  OrderedDict([('alpha', 0.95), ('max_depth', 6), ('max_features', 0.3903246262599702), ('min_impurity_decrease', 0.5), ('min_samples_leaf', 1), ('min_samples_split', 60), ('subsample', 1.0)]) 

# gbr2 = opt.best_estimator_
# gbr2


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



evaluate_model(model_type='regression', model=model, X=X_train_trans, y_true=Y_train)
#evaluate_model(model_type='regression', model=model, X=X_train_trans_reduce, y_true=Y_train)

evaluate_model(model_type='regression', model=model, X=X_test_trans, y_true=Y_test)


# 4.5 Build a Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import VarianceThreshold

pipeline = Pipeline([
    ('col_trans',column_trans),
    # ('std_scalar', StandardScaler()),
    # ('power_trans',PowerTransformer()),
    ('reduce_var',VarianceThreshold(0.01)),
    ('model',model)
])

results = cross_val_score(pipeline, X_train, Y_train, cv=5)
print(results.mean())


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


#df_val['predicted']=np.expm1(model.predict(independent_trans_val))
df_val['SalePrice']=model.predict(independent_trans_val)

#output
out=df_val['SalePrice']
out.to_csv(path_or_buf=full_path+"\\result.csv",index=True)



