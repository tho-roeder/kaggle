# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 14:08:21 2021

@author: thoma
"""

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



nominal=['MSSubClass','MSZoning','Street', 'Alley','LandContour','Utilities','LotConfig','Neighborhood',
         'Condition1','Condition2',
         'BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation',
         'Heating','CentralAir','Electrical','GarageType','PavedDrive','Fence','MiscFeature','SaleType',
         'SaleCondition',
         # 'f_Alley','f_Street','f_MiscFeature','f_FireplaceQu','f_GarageQual','f_Fence'
         ]
categorial=['LotShape','LandSlope','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
            'BsmtFinType2','HeatingQC','KitchenQual','Functional','FireplaceQu','GarageFinish',
            'GarageQual','GarageCond','PoolQC']


category_array=[
    ['Reg','IR1', 'IR2', 'IR3','NA'],
    ['Gtl','Mod','Sev','NA'],
    ['Ex','Gd','TA','Fa','Po','NA'],
    ['Ex','Gd','TA','Fa','Po'],
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
    ['Ex','Gd','TA','Fa','NA']]


# category_array=[
#     ['Reg','IR1', 'IR2', 'IR3'],
#     ['Gtl','Mod','Sev'],
#     ['Ex','Gd','TA','Fa','Po'],
#     ['Ex','Gd','TA','Fa','Po'],
#     ['Ex','Gd','TA','Fa','Po','NA'],
#     ['Ex','Gd','TA','Fa','Po','NA'],
#     ['Gd', 'Av', 'Mn', 'No','NA'],
#     ['GLQ', 'ALQ', 'BLQ','Rec', 'LwQ', 'Unf', 'NA'],
#     ['GLQ', 'ALQ', 'BLQ','Rec', 'LwQ', 'Unf', 'NA'],
#     ['Ex','Gd','TA','Fa','Po'],
#     ['Ex','Gd','TA','Fa','Po'],
#     ['Typ','Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal'],
#     ['Ex','Gd','TA','Fa','Po','NA'],
#     ['Fin','RFn','Unf','NA'],
#     ['Ex','Gd','TA','Fa','Po','NA'],
#     ['Ex','Gd','TA','Fa','Po','NA'],
#     ['Ex','Gd','TA','Fa','NA']]


skew_var=determine_skewed_var(df=df_stacked[df_stacked['f_train']==1]
                              ,num_var=num_var_nan+num_var_nonan
                              ,factor=0.5)
skew_var.remove('SalePrice')


from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer 
# from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PowerTransformer

from sklearn_pandas import DataFrameMapper

num_var_nonan.remove('SalePrice')
mapper = DataFrameMapper(
    [([item],[SimpleImputer(strategy="constant",fill_value='NA')]) for item in str_var_nan]+ 
    [([item],[SimpleImputer(strategy="constant",fill_value='NA')]) for item in str_var_nonan]+
    [([item],[SimpleImputer(strategy='median')]) for item in num_var_nan]+
    [([item],[SimpleImputer(strategy='median')]) for item in num_var_nonan]
    ,df_out =True)


column_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), nominal),
    (OrdinalEncoder(categories=category_array), categorial),
    (PowerTransformer(), skew_var),
    remainder='passthrough',
    verbose=True)



#https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
# make it a pipeline

df_stacked_train= column_trans.fit_transform(df_stacked[df_stacked['f_train']==1])

# 4.5 Build a Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
model = XGBRegressor(objective='reg:gamma'
                      ,booster = 'dart'
                      ,learning_rate=0.2710320608415172
                      ,max_depth=5
                      ,n_estimators=2081
                      )


pipeline = make_pipeline(
    (mapper)
    ,(column_trans)
    ,(StandardScaler())
    ,(VarianceThreshold(0.01))
    ,(model)
)



# dependent=df_train['SalePrice']
# independent=df_train.drop(columns='SalePrice',axis='columns')
# independent_trans=pipeline.fit_transform(independent)
pipe=pipeline.fit(X_train,Y_train)

# X_test['Predict_SalePrice']=pipe.predict(X_test)

evaluate_model(model_type='regression'
                ,model=pipe
                ,X=X_test
                ,y_true=Y_test
                )


# def evaluation(pipeline, X, y):
#     y_predict_proba = pipeline.predict_proba(X)[:, 1]
#     return{
#         'auc': roc_auc_score(y, y_predict_proba)
#     }




# evaluation(pipeline, X, y)

from sklearn.metrics import mean_squared_error, r2_score
  
#Test
print('Mean squared error Test: %.2f'
      % mean_squared_error(df_test['SalePrice'], pipe_test))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination R2 Test: %.2f'
      % r2_score(df_train['SalePrice'], pipe_test))


best_model=pre_eval_models(
    type_model='regression',
    dependent=independent_trans, 
    inpendent=dependent,
    scoring='neg_root_mean_squared_error',
    cv=10)



# 3.5 check for missing
#num_var_nonan.remove('SalePrice')
print(df[str_var_nan + str_var_nonan].isna().sum())
print(df[num_var_nan + num_var_nonan].isna().sum())

# new=pipe_test.isna().sum()


# 3.6 enrich data:
# Variable treement String:  
str_lst_flags=['Alley','Street','MiscFeature','FireplaceQu','GarageQual','Fence']
for i in str_lst_flags:
    df_stacked['f_'+str(i)]=df_stacked[i].apply(lambda x: create_flags(x))

# feture engineering a new feature "TotalSF"
df_stacked['TotalSF'] = df_stacked['TotalBsmtSF'] + df_stacked['1stFlrSF'] + df_stacked['2ndFlrSF']
df_stacked['Total_Bathrooms'] = (df_stacked['FullBath'] + (0.5 * df_stacked['HalfBath']) + df_stacked['BsmtFullBath'] + (0.5 * df_stacked['BsmtHalfBath']))




