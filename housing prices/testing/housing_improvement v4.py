# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 14:08:21 2021

@author: thoma
"""

# https://ryankresse.com/convenient-preprocessing-with-sklearn_pandas-dataframemapper/
# https://rittikghosh.com/PolynomialFeatures.html
# https://github.com/jpmml/jpmml-sklearn/blob/master/src/test/resources/main.py
#https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html


#http://numba.pydata.org/
#https://numba.readthedocs.io/en/stable/roc/kernels.html#introduction

### 1. import helper
import os
path="C:\\Users\\thoma\\Desktop\\VM share\\Python\\Kaggle\\helper"
os.chdir(path)
%run helper.py


### 2. source data
full_path="C:\\Users\\thoma\\Desktop\\VM share\\Python\\files\\Kaggle\\housing prices"
import pandas as pd
df_train=pd.read_csv(full_path+"\\train.csv", index_col='Id')
df_test=pd.read_csv(full_path+"\\test.csv", index_col='Id')


# For identification purposes
df_train["f_train"] = 1
df_test["f_train"] = 0
df_test["SalePrice"] = 0
df_stacked = pd.concat([df_train, df_test])


### 3. exploratory data analysis
# 3.1 create overview of data set:
#getGeneralInformation(df)
all_var, num_var_nan, num_var_nonan, str_var_nan, str_var_nonan= pre_work(df_stacked)

num_var_nonan.remove('SalePrice')

# 3.2 plot variables:
# plot_str_var(df=df_stacked[df_stacked['f_train']==1],var=str_var_nan)
# plot_str_var(df=df_stacked[df_stacked['f_train']==1],var=['Street'])
# plot_num_var(df=df_stacked[df_stacked['f_train']==1],var=num_var_nan)

# 3.3 corr
df_high_corr, high_corr_var=get_df_high_corr_target(df=df_stacked[df_stacked['f_train']==1],target='SalePrice',min_cor=0.5)
# get_scatter_for_target(df=df_stacked[df_stacked['f_train']==1],var=high_corr_var,target='SalePrice')

# 3.4 impute without data leakage

#drop_list_str,impute_value_str=impute_var_v54df=df_stacked[df_stacked['f_train']==1],var=str_var_nan,perc_drop=1,style='value',value='NA')

# from sklearn.impute import SimpleImputer
# imp = SimpleImputer(strategy="constant",fill_value='NA')
# print(imp.fit_transform(df_stacked[df_stacked['f_train']==1][str_var_nan]))
# returned=imp.transform(df_stacked[df_stacked['f_train']==0][str_var_nan])

# returned_df = pd.DataFrame(returned, columns = df_stacked[str_var_nan].columns)

# X_selected_df = pd.DataFrame(X_selected, columns=[X_train.columns[i] for i in range(len(X_train.columns)) if feature_selector.get_support()[i]])


### 4 pred data
# 4.1 split dependent and independent
dependent=df_train['SalePrice']
independent=df_train.copy(deep=True)
independent.drop(columns='SalePrice',axis='columns', inplace=True)


# 4.2.3 split test train data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(independent,dependent,test_size=0.3, random_state=41, shuffle=True)

import numpy as np
Y_train_Log1p = np.log1p(Y_train)
Y_test_Log1p = np.log1p(Y_test)


################################
# 4.5 Build a Pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_union
from sklearn_pandas import DataFrameMapper

from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import TransformerMixin


from xgboost import XGBRegressor

################################

class AddVariablesNotImputed(TransformerMixin):

    def __init__(self):
        return None
    
    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        str_lst_flags=['Alley','Street','MiscFeature','FireplaceQu','GarageQual','Fence']
        for i in str_lst_flags:
            X['f_'+str(i)]=X[i].apply(lambda x: create_flags(x))
        return X

################################

impute = DataFrameMapper(
      [([item],SimpleImputer(strategy="constant",fill_value='NA')) for item in str_var_nan]
    + [([item],SimpleImputer(strategy="constant",fill_value='NA')) for item in str_var_nonan]
    + [([item],SimpleImputer(strategy='median')) for item in num_var_nan]
    + [([item],SimpleImputer(strategy='median')) for item in num_var_nonan]
    # ,default=None
    ,df_out =True)

################################

class AddVariablesImputed(TransformerMixin):

    def __init__(self):
        return None
    
    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF']+ X['2ndFlrSF']
        X['Total_Bathrooms'] = (X['FullBath'] 
                                         + (0.5 * X['HalfBath']) 
                                         + X['BsmtFullBath'] 
                                         + (0.5 * X['BsmtHalfBath']))
        
        X['Age_Garage']=X['YrSold']-X['GarageYrBlt']
        X['Age_House']=X['YrSold']-X['YearBuilt']
        X['Age_Remod']=X['YrSold']-X['YearRemodAdd']

        poly= [
            ['LotArea','LotShape'],
            ['OverallQual','OverallCond'],
            ['ExterQual','ExterCond'],
            ['BsmtQual','BsmtCond'],
            ['BsmtFinType1','BsmtFinSF1'],
            ['BsmtFinType2','BsmtFinSF2'],
            # ['GarageFinish','GarageArea','GarageQual','GarageCond'],
            ['GarageFinish','GarageArea'],
            ['GarageQual','GarageCond'],
            ['Fireplaces','FireplaceQu'],
            ['PoolArea','PoolQC'],
            # ['Kitchen','KitchenQual']
            ]
        
        for item in poly:
            X['i_'+str(item[0]+str(item[1]))]=X[item[0]] * X[item[1]]
            
        return X

################################

nominal=['MSSubClass','MSZoning','Street', 'Alley','LandContour','Utilities','LotConfig','Neighborhood',
         'Condition1','Condition2',
         'BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation',
         'Heating','CentralAir','Electrical','GarageType','PavedDrive','Fence','MiscFeature','SaleType',
         'SaleCondition'
         # ,'f_Alley','f_Street','f_MiscFeature','f_FireplaceQu','f_GarageQual','f_Fence'
         ]
categorial=['LotShape','LandSlope','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
            'BsmtFinType2','HeatingQC','KitchenQual','Functional','FireplaceQu','GarageFinish',
            'GarageQual','GarageCond','PoolQC']

category_array=[
    ['Reg','IR1', 'IR2', 'IR3','NA'],
    ['Gtl','Mod','Sev','NA'],
    ['Ex','Gd','TA','Fa','Po','NA'],
    ['Ex','Gd','TA','Fa','Po','NA'],
    ['Ex','Gd','TA','Fa','Po','NA'],
    ['Ex','Gd','TA','Fa','Po','NA'],
    ['Gd', 'Av', 'Mn', 'No','NA'],
    ['GLQ', 'ALQ', 'BLQ','Rec', 'LwQ', 'Unf', 'NA'],
    ['GLQ', 'ALQ', 'BLQ','Rec', 'LwQ', 'Unf', 'NA'],
    ['Ex','Gd','TA','Fa','Po','NA'],
    ['Ex','Gd','TA','Fa','Po','NA'],
    ['Typ','Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal','NA'],
    ['Ex','Gd','TA','Fa','Po','NA'],
    ['Fin','RFn','Unf','NA'],
    ['Ex','Gd','TA','Fa','Po','NA'],
    ['Ex','Gd','TA','Fa','Po','NA'],
    ['Ex','Gd','TA','Fa','NA']
    ]

column_to_cat=dict(zip(categorial,category_array))

skew_var=determine_skewed_var(df=X_train
                              ,num_var=num_var_nan+num_var_nonan
                              ,factor=0.5)

transformation = DataFrameMapper(
      [([item],[OneHotEncoder(handle_unknown='ignore')]) for item in nominal]
    + [([col],OrdinalEncoder(categories = [cat])) for col, cat in column_to_cat.items()]
    + [([item],PowerTransformer()) for item in skew_var]
    # + [item for item in remainder]
    # + [([item],ColumnExtractor(item)) for item in remainder]
    ,default=None
    ,df_out=True
)

################################

#add PolynomialFeatures

# poly= [['LotArea','LotShape'],
#  ['OverallQual','OverallCond'],
#  ['ExterQual','ExterCond'],
#  ['BsmtQual','BsmtCond'],
#  ['BsmtFinType1','BsmtFinSF1'],
#  ['BsmtFinType2','BsmtFinSF2'],
#  # ['GarageFinish','GarageArea','GarageQual','GarageCond'],
#  ['GarageQual','GarageCond'],
#  ['Fireplaces','FireplaceQu'],
#  ['PoolArea','PoolQC'],
#  ['Kitchen','KitchenQual']
#  ]

# poly=[['LotFrontage','LotArea']
#       ,['OverallQual','OverallCond']
#       ]

# from sklearn.base import TransformerMixin
# class SelfPolyfeature(TransformerMixin):

#     def __init__(self, transformer_list):
#         self.transformer_list = transformer_list
    
#     def fit(self, X, y=None):
#         self.pf = PolynomialFeatures(interaction_only=True)
#         for i in self.transformer_list:
#             self.pf.fit(X[i])
#         return self

#     def transform(self, X):
#         out = [self.pf.transform(X[i]) for i in self.transformer_list]
#         return out



# from sklearn.base import TransformerMixin
# class SelfPolyfeature(TransformerMixin):

#     def __init__(self, transformer_list):
#         self.transformer_list = transformer_list
    
#     def fit(self, X, y=None):
#         self.pf = PolynomialFeatures(interaction_only=True)
#         return self

#     def transform(self, X):
#         import pandas as pd
#         out= [pd.DataFrame(self.pf.transform(X[i])) for i in self.transformer_list]
#         return out


# polytTest=SelfPolyfeature(poly)

# polytest=polytTest.fit(df_train)

# poly=make_union(
#     (ColumnExtractor([])
#     ,(PolynomialFeatures(interaction_only=True))
#     )


# ft=PolynomialFeatures(interaction_only=True)


# test=ft.fit_transform(independent_trans[['LotFrontage','LotArea']])
# test=ft.fit_transform(independent_trans[['LotFrontage','LotArea']])[:,3]


# all_columns =independent_trans.columns


poly=['LotFrontage','LotArea']

mapper_df2 = DataFrameMapper(
      [([item],PolynomialFeatures(interaction_only=True)) for item in poly]
    ,default=None
    ,df_out=True
)


class ColumnExtractor(TransformerMixin):

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xcols = X[self.cols]
        return Xcols


var_list= ['LotArea','LotShape']
  # ['ExterQual','ExterCond'],
  # ['BsmtQual','BsmtCond'],
  # ['BsmtFinType1','BsmtFinSF1'],
  # ['BsmtFinType2','BsmtFinSF2'],
  # # ['GarageFinish','GarageArea','GarageQual','GarageCond'],
  # ['GarageQual','GarageCond'],
  # ['Fireplaces','FireplaceQu'],
  # ['PoolArea','PoolQC'],
  # ['Kitchen','KitchenQual']
  # ]


# poly=make_union(
#         make_pipeline(
#             (ColumnExtractor(var_list))
#             ,(PolynomialFeatures(interaction_only=True))
#             ,(VarianceThreshold())
#         )
#     )



poly=make_pipeline(
            (ColumnExtractor(var_list))
            ,(PolynomialFeatures(interaction_only=True))
            ,(VarianceThreshold())
        )


################################

pipeline_pre = make_pipeline(
    AddVariablesNotImputed()
    ,(impute)
    ,(transformation)
    ,AddVariablesImputed()
    ,(VarianceThreshold(0.1))
    ,(StandardScaler())
)

independent_trans=pipeline_pre.fit_transform(independent)

best_model=pre_eval_models(
    type_model='regression',
    independent=independent_trans, 
    dependent=dependent,
    scoring='neg_root_mean_squared_error',
    cv=10)

################################

# pipeline = make_pipeline(
#     ,AddVariablesNotImputed()
#     (impute)
#     ,AddVariablesImputed()
#     ,(transformation)
#     # ,poly
#     # ,(PolynomialFeatures(interaction_only=True))
#     # ,(mapper_df2)
#     # ,(SelfPolyfeature(poly))
#     ,(VarianceThreshold(0.01))
#     ,(StandardScaler())
#     ,(XGBRegressor())
# )


# # 4.3.2 RandomizedSearchCV
# from sklearn.model_selection import RandomizedSearchCV
# params={
#         'xgbregressor__objective': ['reg:gamma', 'reg:squarederror', 'reg:squaredlogerror']
#         ,'xgbregressor__booster': ['gbtree','gblinear','dart']
#         ,'xgbregressor__learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.15, 0.2, 0.25, 0.3]
#         ,'xgbregressor__n_estimators': [50,100,200,500,1000,2000,5000]
#         ,'xgbregressor__max_depth': [3,7,11,12,13,15,30]
#         }

# trans_model =RandomizedSearchCV(pipeline
#                           ,params
#                           ,scoring='neg_root_mean_squared_error'
#                           ,cv = 2
#                           ,n_jobs=-1)
# trans_model.fit(X_train, Y_train)



pipeline = make_pipeline(
    AddVariablesNotImputed()
    ,(impute)
    ,(transformation)
    ,AddVariablesImputed()
    # ,poly
    # ,(PolynomialFeatures(interaction_only=True))
    # ,(mapper_df2)
    # ,(SelfPolyfeature(poly))
    ,(VarianceThreshold(0.1))
    ,(StandardScaler())
)

X_train_trans=pipeline.fit_transform(X_train.copy())


# 4.3.2 RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
params={
        'objective': ['reg:gamma', 'reg:squarederror', 'reg:squaredlogerror']
        ,'booster': ['gbtree','gblinear','dart']
        ,'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.15, 0.2, 0.25, 0.3]
        ,'n_estimators': [50,100,200,500,1000,2000,5000]
        ,'max_depth': [3,7,11,12,13,15,30]
        }

from sklearn.model_selection import StratifiedKFold

trans_model =RandomizedSearchCV(XGBRegressor()
                          ,params
                          ,scoring='neg_root_mean_squared_error'
                          ,cv = 3
                          #StratifiedKFold(n_splits=2,shuffle=True,random_state=41)
                          ,n_jobs=-1)


trans_model.fit(X_train_trans, Y_train)


# pipeline.get_params().keys()

for param, score in zip(trans_model.cv_results_['params'], trans_model.cv_results_['mean_test_score']):
    print (param, score)
print(trans_model.best_params_)
# {'xgbregressor__objective': 'reg:gamma', 'xgbregressor__n_estimators': 1000, 'xgbregressor__max_depth': 15, 'xgbregressor__learning_rate': 0.25, 'xgbregressor__booster': 'dart'} -28500.581839438946
# {'xgbregressor__objective': 'reg:gamma', 'xgbregressor__n_estimators': 5000, 'xgbregressor__max_depth': 15, 'xgbregressor__learning_rate': 0.2, 'xgbregressor__booster': 'dart'} -27726.567609741913


# 4.4 Check metrics
X_test_trans=pipeline.transform(X_test.copy())
evaluate_model(model_type='regression'
                ,model=trans_model
                ,X=X_test_trans
                ,y_true=Y_test
                )


# 5. Apply model
# apply model to validation
#df_test['predicted']=np.expm1(trans_model.predict(df_test))
df_test['SalePrice']=trans_model.predict(pipeline.transform(df_test))

#output
out=df_test['SalePrice']
out.to_csv(path_or_buf=full_path+"\\result.csv",index=True)





