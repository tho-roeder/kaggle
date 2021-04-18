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

#feature selection/feature importance

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

df_stacked_train=df_train.copy()
df_stacked_test=df_test.copy()

# For identification purposes
df_stacked_train["f_train"] = 1
df_stacked_test["f_train"] = 0
df_stacked_test["SalePrice"] = 0
df_stacked = pd.concat([df_stacked_train, df_stacked_test])


### 3. exploratory data analysis
# 3.1 create overview of data set:
#getGeneralInformation(df)
all_var, num_var_nan, num_var_nonan, str_var_nan, str_var_nonan= pre_work(df_stacked)

num_var_nonan.remove('SalePrice')
num_var_nonan.remove('f_train')
all_var.remove('f_train')

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
        #here issue
        # str_lst_flags=['Alley','Street','MiscFeature','FireplaceQu','GarageQual','Fence']
        # for i in str_lst_flags:
        #     X['f_'+str(i)]=X[i].apply(lambda x: create_flags(x))
        
        other_lst_flags=['PoolArea', '2ndFlrSF','TotalBsmtSF','Fireplaces','GarageArea','LowQualFinSF'#,'TotalPorch'
                          ,'MiscVal','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch']
        for i in other_lst_flags:
            X[i]=X[i].fillna(0)
            X['f_'+str(i)]=X[i].apply(lambda x: 0 if x == 0 else 1)
        

        
        #############
        # dff["TotalPorchType"] = dff["HasWoodDeck"] + dff["HasOpenPorch"] + dff["HasEnclosedPorch"] + dff["Has3SsnPorch"] + dff["HasScreenPorch"]
        # dff["TotalPorchType"] = dff["TotalPorchType"].apply(lambda x: 3 if x >=3 else x)
        
        return X

################################

impute = DataFrameMapper(
      [([item],SimpleImputer(strategy="constant",fill_value='NA')) for item in str_var_nan]
    + [([item],SimpleImputer(strategy="constant",fill_value='NA')) for item in str_var_nonan]
    + [([item],SimpleImputer(strategy='median')) for item in num_var_nan]
    + [([item],SimpleImputer(strategy='median')) for item in num_var_nonan]
    ,default=None
    ,df_out =True)

################################


class AddVariablesImputed(TransformerMixin):

    def __init__(self):
        return None
    
    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        X["f_North"] = X["Neighborhood"].apply(lambda x: 1 if x in ["Blmngtn", "BrDale", "ClearCr", "Gilbert",  "Names", "NoRidge","NPkVill", "NWAmes", "NoRidge", "NridgHt", "Sawyer", "Somerst", "StoneBr", "Veenker", "NridgHt"] else 0)
        X["f_South"] = X["Neighborhood"].apply(lambda x: 1 if x in ["Blueste", "Edwards", "Mitchel", "MeadowV", "SWISU", "IDOTRR", "Timber"] else 0)
        X["f_East"] = X["Neighborhood"].apply(lambda x: 1 if x in ["IDOTRR", "Mitchel"] else 0)
        X["f_West"] = X["Neighborhood"].apply(lambda x: 1 if x in ["Edwards", "NWAmes", "SWISU", "Sawyer", "SawyerW"] else 0)
        X["f_Downtown"] = X["Neighborhood"].apply(lambda x: 1 if x in ["BrkSide", "Crawfor", "OldTown", "CollgCr"] else 0)
        
                
        # dff["TotalPorchType"] = dff["HasWoodDeck"] + dff["HasOpenPorch"] + dff["HasEnclosedPorch"] + dff["Has3SsnPorch"] + dff["HasScreenPorch"]
        # dff["TotalPorchType"] = dff["TotalPorchType"].apply(lambda x: 3 if x >=3 else x)
                
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

class AddVariablesTransformed(TransformerMixin):

    def __init__(self):
        return None
    
    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        #GarageValue
        #BsmtValue
        
        
        # dff["TotalValue"] = dff["OverallValue"] + dff["ExterValue"] + dff["BsmtValue"] + dff["KitchenValue"] + dff["FireplaceValue"] + dff["GarageValue"] +\
        # dff["HeatingQC"] + dff["Utilities"] + dff["Electrical"] - dff["Functional"]  + dff["PoolQC"]
        
        # dff["TotalAreaXOverallValue"] = dff["TotalArea"] * dff["OverallValue"]
        # dff["TotalAreaXOverallQual"] = dff["TotalArea"] * dff["OverallQual"]
        # dff["BsmtValue"] = ((dff["BsmtQual"] + dff["BsmtFinType1"] + dff["BsmtFinType2"]) * dff["BsmtCond"]) / 2
        
        # in second trans
        # dff["WeightedBsmtFinSF1"] = dff["BsmtFinSF1"] * dff["BsmtFinType1"]
        # dff["WeightedBsmtFinSF2"] = dff["BsmtFinSF2"] * dff["BsmtFinType2"]
        # dff["WeightedTotalBasement"] =  dff["WeightedBsmtFinSF1"] + dff["BsmtFinSF2"] * dff["BsmtFinType2"] +  dff["BsmtUnfSF"]
       
        
        #####
        X['TotalQual'] = X['OverallQual'] + X['ExterQual'] + X['BsmtQual'] + X['KitchenQual'] + X['FireplaceQu'] + X['GarageQual'] + X['HeatingQC'] + X['PoolQC']
        X['TotalCond'] = X['OverallCond'] + X['ExterCond'] + X['BsmtCond'] + X['GarageCond']
        X['TotalQualCond'] = X['TotalQual'] + X['TotalCond']
        
        X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF']+ X['2ndFlrSF']
        X['HQFloor'] = X['1stFlrSF'] + X['2ndFlrSF']
        
        X['Total_Bathrooms'] = (X['FullBath'] + (0.5 * X['HalfBath']) + X['BsmtFullBath'] + (0.5 * X['BsmtHalfBath']))
        X["TotalFullBath"] = X["BsmtFullBath"] + X["FullBath"]
        X["TotalHalfBath"] = X["BsmtHalfBath"] + X["HalfBath"]
        X["TotalBsmtBath"] = X["BsmtFullBath"] + 0.5 * X["BsmtHalfBath"]
        X["TotalBath"] = X["FullBath"] + 0.5 * X["HalfBath"]
        
        
        X["TotalArea"] = X["TotalBsmtSF"] + X["GrLivArea"]
        X["TotalPorch"] = X["WoodDeckSF"] + X["OpenPorchSF"] + X["EnclosedPorch"] + X["3SsnPorch"] + X["ScreenPorch"]

        X['FrontageRatio'] = (X['LotFrontage'] / X['LotArea'])

        #Ages
        X['Age_Garage']=X['YrSold']-X['GarageYrBlt']
        X['Age_House']=X['YrSold']-X['YearBuilt']
        X['Age_House'] = X['Age_House'].apply(lambda x: 0 if x < 0 else x)
        X['f_NewHouse'] = X['Age_House'].apply(lambda x: 1 if x == 0 else 0)

        X['Age_SinceRemod']=X['YrSold']-X['YearRemodAdd']
        X['f_SinceRemod'] = X['Age_SinceRemod'].apply(lambda x: 0 if x < 0 else x)
        
        X['Age_RemodAfter']=X['YearRemodAdd']-X['YearBuilt']
        X['Age_RemodAfter']=X['Age_RemodAfter'].apply(lambda x: 0 if x < 0 else x)
        X['f_RemodAfter'] = X['Age_RemodAfter'].apply(lambda x: 0 if x == 0 else 1)

        #polynomial
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
            ['KitchenAbvGr','KitchenQual'],
            # ['TotalBsmtSF', 'BsmtValue'],
            # ['TotalBsmtSF','BsmtValue'],
            ['TotalBsmtSF','BsmtQual'],
            # ['GarageArea','GarageValue'],
            ['GarageArea','GarageQual']
            ]
        

        for item in poly:
            X['i_'+str(item[0]+str(item[1]))]=X[item[0]] * X[item[1]]
        
        return X


################################

pipeline_pre = make_pipeline(
    AddVariablesNotImputed()
    ,(impute)
    ,AddVariablesImputed()
    ,(transformation)
    ,AddVariablesTransformed()
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
    ,AddVariablesImputed()
    ,(transformation)
    ,AddVariablesTransformed()
    ,(VarianceThreshold(0.01))
    ,(StandardScaler())
)

X_train_trans=pipeline.fit_transform(X_train.copy())
X_test_trans=pipeline.transform(X_test.copy())

# 4.3.2 RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV

params={
        'objective': ['reg:gamma', 'reg:squarederror', 'reg:squaredlogerror']
        ,'booster': ['gbtree','gblinear','dart']
        ,'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.15, 0.2, 0.25, 0.3]
        ,'n_estimators': [50,100,200,500,1000,2000,5000]
        ,'max_depth': [3,7,11,12,13,15,30]
        }

# the_model = XGB.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
#                              learning_rate=0.05, max_depth=3, 
#                              min_child_weight=1.7817, n_estimators=2200,
#                              reg_alpha=0.4640, reg_lambda=0.8571,
#                              subsample=0.5213, random_state =7, nthread = -1)



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

# {'objective': 'reg:gamma', 'n_estimators': 5000, 'max_depth': 11, 'learning_rate': 0.1, 'booster': 'gbtree'} -30769.86281498698

# 4.4 Check metrics

evaluate_model(model_type='regression'
                ,model=trans_model
                ,X=X_test_trans
                ,y_true=Y_test
                )

# w Variance w StandardScaler                                Negative RMSLE -0.13198577792361038 public: 0.14074
# w/o Variance (0.1)                                         Negative RMSLE -0.12639349257885119 public: 0.13975
# w/o Variance (0.1) w/o StandardScaler                      Negative RMSLE -0.13382081095432172 public: 0.14428
# w Log Y                                                    Negative RMSLE -0.01024387506624063 public: 0.14504
# w Variance (0.1) w StandardScaler + new var                Negative RMSLE -0.12724941175938065 public: 0.14137
# w Variance (0.1) w StandardScaler + new var add            Negative RMSLE -0.15808536930192083
# w Variance (0.01) w StandardScaler + new var add           Negative RMSLE -0.1316752633106005  public: 0.13033
# w Variance () w StandardScaler + new var add -f_fence      Negative RMSLE -0.12657911346085915 public: 0.13769
# w Variance (0.01) w StandardScaler + new var add           Negative RMSLE -0.12975623955885035
    #{'objective': 'reg:gamma', 'n_estimators': 2000, 'max_depth': 30, 'learning_rate': 0.15, 'booster': 'dart'}


# 5. Apply model
# apply model to validation

df_test=pd.read_csv(full_path+"\\test.csv", index_col='Id')
df_test_trans=pipeline.transform(df_test.copy())

#df_test['SalePrice']=np.expm1(trans_model.predict(df_test_trans))
df_test['SalePrice']=trans_model.predict(df_test_trans)

#output
out=df_test['SalePrice']
out.to_csv(path_or_buf=full_path+"\\result.csv",index=True)





