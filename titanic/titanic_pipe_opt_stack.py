# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 11:38:20 2021

@author: thoma
"""

### 1. import helper
import os
path="C:\\Users\\thoma\\Desktop\\VM share\\Python\\Kaggle\\helper"
os.chdir(path)
%run helper.py

### 2. source data

full_path="C:\\Users\\thoma\\Desktop\\VM share\\Python\\files\\Kaggle\\titanic"

import pandas as pd
df_train=pd.read_csv(full_path+"\\train.csv",index_col='PassengerId')
df_test=pd.read_csv(full_path+"\\test.csv",index_col='PassengerId')

df_stacked_train=df_train.copy()
df_stacked_test=df_test.copy()

# For identification purposes
df_stacked_train["f_train"] = 1
df_stacked_test["f_train"] = 0
df_stacked_test["Survived"] = 0
df_stacked = pd.concat([df_stacked_train, df_stacked_test])


### 3. exploratory data analysis (EDA)
# 3.1 create overview of data set:
#getGeneralInformation(df)
all_var, num_var_nan, num_var_nonan, str_var_nan, str_var_nonan= pre_work(df_stacked)

num_var_nonan.remove('Survived')
num_var_nonan.remove('f_train')
all_var.remove('f_train')

# 3.2 plot variables:
# plot_str_var(df=df_stacked[df_stacked['f_train']==1],var=str_var_nan)
# plot_str_var(df=df_stacked[df_stacked['f_train']==1],var=['Street'])
# plot_num_var(df=df_stacked[df_stacked['f_train']==1],var=num_var_nan)

# 3.3 corr
df_high_corr, high_corr_var=get_df_high_corr_target(df=df_stacked[df_stacked['f_train']==1],target='Survived',min_cor=0.5)
# get_scatter_for_target(df=df_stacked[df_stacked['f_train']==1],var=high_corr_var,target='Survived')

# 3.4 impute without data leakage
# drop_list_str,impute_value_str=impute_var_v4(df=df,var=str_var_nan,perc_drop=1,style='value',value='NA')


### 4 pred data
# 4.1 split dependent and independent
dependent=df_train['Survived']
independent=df_train.copy(deep=True)
independent.drop(columns='Survived',axis='columns', inplace=True)

# 4.2 split test train data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(independent,dependent,test_size=0.2, random_state=41, shuffle=True)

# 4.3 Build a Pipeline
# 4.3.1 Lib import
# general
import numpy as np

#Pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_union
from sklearn_pandas import DataFrameMapper

#Imputation & Transformation
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import TransformerMixin

#Feature Selection
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.feature_selection import SequentialFeatureSelector

#Model 
from catboost import CatBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

#Stacking
from mlxtend.regressor import StackingRegressor 
#from mlxtend.regressor import StackingCVRegressor
from sklearn.ensemble import StackingRegressor as stack


# 4.3.2 pipeline steps
class AddVariablesNotImputed(TransformerMixin):

    def __init__(self):
        return None
    
    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):

        str_lst_flags=['Cabin', 'Ticket']
        for i in str_lst_flags:
            X['f_'+str(i)]=X[i].apply(lambda x: create_flags(x))
            
           
        # other_lst_flags=['Cabin', 'Ticket']
        # for i in other_lst_flags:
        #     X[i]=X[i].fillna(0)
        #     X['f_'+str(i)]=X[i].apply(lambda x: 0 if x == 0 else 1)
      
        return X


#############

impute = DataFrameMapper(
      [([item],SimpleImputer(strategy="constant",fill_value='NA')) for item in str_var_nan]
    + [([item],SimpleImputer(strategy="constant",fill_value='NA')) for item in str_var_nonan]
    + [([item],SimpleImputer(strategy='median')) for item in num_var_nan]
    + [([item],SimpleImputer(strategy='median')) for item in num_var_nonan]
    ,drop_cols=None
    ,default=None
    ,df_out =True
    )

#############
  #name 
        
class AddVariablesImputed(TransformerMixin):

    def __init__(self):
        return None
    
    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        X['TravelAlone']=np.where((X["SibSp"]+X["Parch"])>0, 0, 1)
        X['f_SibSp']=np.where((X["SibSp"])>0, 1, 0)
        X['f_Parch']=np.where((X["Parch"])>0, 1, 0)
        X['IsMinor']=np.where(X['Age']<=16, 1, 0)
        
        
        X['Family_Size']=X["SibSp"]+X["Parch"]             
        # dff["TotalPorchType"] = dff["HasWoodDeck"] + dff["HasOpenPorch"] + dff["HasEnclosedPorch"] + dff["Has3SsnPorch"] + dff["HasScreenPorch"]
        # dff["TotalPorchType"] = dff["TotalPorchType"].apply(lambda x: 3 if x >=3 else x)
                
        return X

#############
# df=pd.get_dummies(df,columns=["Pclass","Embarked","Sex","Cabin_new","Age_bin","Fare_bin","Title","Boarded"])

drop=['Cabin','Name','Ticket']

nominal=['Embarked','Sex','Pclass','f_Cabin','f_Ticket']

# categorial=[]
# category_array=[]

# column_to_cat=dict(zip(categorial,category_array))

# skew_var=determine_skewed_var(df=X_train
#                               ,num_var=num_var_nan+num_var_nonan
#                               ,factor=0.5)

transformation = DataFrameMapper(
    [([item],OneHotEncoder(handle_unknown='ignore')) for item in nominal]
    # +[([item],None) for item in pass_through]
    # + [([col],OrdinalEncoder(categories = [cat])) for col, cat in column_to_cat.items()]
    # + [([item],PowerTransformer()) for item in skew_var]
    ,default=None #None #False
    ,df_out=True
    , drop_cols=drop
)

################################

pipeline_pre = make_pipeline(
    AddVariablesNotImputed()
    ,(impute)
    ,AddVariablesImputed()
    ,(transformation)
    # ,AddVariablesTransformed()
    ,(VarianceThreshold(0.1))
    ,(StandardScaler())
)

independent_trans=pipeline_pre.fit_transform(independent.copy())

best_model=pre_eval_models(
    type_model='classification',
    independent=independent_trans, 
    dependent=dependent,
    scoring='accuracy',
    cv=10)

################################

# # 4.3.2 RandomizedSearchCV

pipeline = make_pipeline(
    AddVariablesNotImputed()
    ,(impute)
    ,AddVariablesImputed()
    ,(transformation)
    # ,AddVariablesTransformed()
    ,(VarianceThreshold(0.1))
    ,(StandardScaler())
    ,GradientBoostingClassifier()
)

trans_model=pipeline.fit(X_train.copy(),Y_train.copy())

# 4.4 Check metrics

evaluate_model(model_type='classification'
                ,model=trans_model
                ,X=X_test.copy()
                ,y_true=Y_test.copy()
                )

# 5. Apply model
# apply model to validation
df_test=pd.read_csv(full_path+"\\test.csv",index_col='PassengerId')

df_test['Survived']=trans_model.predict(df_test.copy())

#output
out=df_test['Survived']
out.to_csv(path_or_buf=full_path+"\\result.csv",index=True)



