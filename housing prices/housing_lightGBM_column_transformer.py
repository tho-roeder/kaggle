# -*- coding: utf-8 -*-

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
#plot_num_var(df=df,var=num_var_nan)
# median for num_var_nan


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


### 4 pred data
# 4.1 
dependent=df['SalePrice']
independent=df.copy(deep=True)
independent.drop(columns='SalePrice',axis='columns', inplace=True)

# 4.2 split test train data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(independent,dependent,test_size=0.3, random_state=41, shuffle=True)

# 4.3 impute variables
# 4.3.1 check for missing
num_var_nonan.remove('SalePrice')
print(X_train[str_var_nan + str_var_nonan].isna().sum())
print(X_train[num_var_nan + num_var_nonan].isna().sum())

# 4.3.2 impute string
drop_list_str,impute_value_str=impute_var_v4(df=X_train,var=str_var_nan,perc_drop=1,style='value',value='NA')
#drop_list_str,impute_value_str=impute_var_v4(df=X_train,var=str_lst_mode,perc_drop=1,style='mode')
drop_list_num,impute_value_num=impute_var_v4(df=X_train,var=num_var_nan,perc_drop=0.25,style='median')

# 4.3.3 dropping variables
drop_list_lowCor=merge_low_corr(df_ind=X_train, df_dep=Y_train, target='SalePrice', min_cor=0.05)
# get_scatter_for_target(df=df,var=drop_list_lowCor,target='SalePrice')

all_var.remove('SalePrice')
drop_list_max_perc_rep=same_value(df=X_train,var=all_var,max_perc_rep=0.95)
# get_violinplot_for_target(df=df,var=drop_list_max_perc_rep,target='SalePrice')


def log_var(df, num_var):
    #import seaborn as sns
    apply_value_log=[]
    from scipy.stats import skew
    skewed_feats = df[num_var].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    for i in skewed_feats.index:
        if skewed_feats[i]>=0.7:
            #   sns.displot(df[i])
            df['log_'+i]=np.log1p(df[i])
            # sns.displot(df['log_'+i])
            apply_value_log.append(i)
    return apply_value_log
            
apply_value_log=log_var(df=X_train, num_var=num_var_nan + num_var_nonan)


# 4.3.4 final list
drop_list=[]
# drop_list.extend(drop_list_num)
# drop_list.extend(drop_list_str)
# drop_list.extend(drop_list_lowCor)
# drop_list.extend(drop_list_max_perc_rep)
# drop_list.extend(str_lst_scale)
# drop_list.extend(str_lst_flags) 

X_train.drop(columns=drop_list,axis='columns', inplace=True)


# 4.3.5 one hot encoding vs. numeric tranformation
# use OrdinalEncoder for categorical and OneHotEncoder for numerical; use pipeline
# do not use lable encoder:https://www.youtube.com/watch?v=0w78CHM_ubM 
# pipline: https://www.youtube.com/watch?v=irHhDMbw3xo&list=PLaAHY_Fq4qw2sXUhe4GHQm7lxtBm38Nzz&index=22

nominal=['MSSubClass','MSZoning','Street', 'Alley','LandContour','Utilities','LotConfig','Neighborhood',
         'Condition1','Condition2',
         'BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation',
         'Heating','CentralAir','Electrical','GarageType','PavedDrive','Fence','MiscFeature','SaleType',
         'SaleCondition',
         'f_Alley','f_Street','f_MiscFeature','f_FireplaceQu','f_GarageQual','f_Fence'
         ]
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
    ['Ex','Gd','TA','Fa','NA']
    ]


#sklearn.preprocessing.OneHotEncoder
#https://towardsdatascience.com/one-hot-encoding-multicollinearity-and-the-dummy-variable-trap-b5840be3c41a
#X_train=pd.get_dummies(X_train)
# OneHotEncoder is prefered as it creates an object that can be applied to train & validation df
# from  sklearn.preprocessing import OneHotEncoder
# oneHot_encoder = OneHotEncoder(sparse=True)
# other=oneHot_encoder.fit_transform(test_001[nominal])
# oneHot_encoder.categories_

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
column_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), nominal),
    (OrdinalEncoder(categories=category_array), categorial),
    remainder='passthrough')
X_train.sort_index(axis=1, inplace=True)
X_train_fitted=column_trans.fit_transform(X_train)
#print(column_trans.named_transformers_.ordinalencoder.categories_)

# 4.4 apply transformation on X_test 
X_test.drop(columns=drop_list,axis='columns', inplace=True)
for i in impute_value_num.keys():
    X_test[i].fillna(impute_value_num[i], inplace=True)
for i in impute_value_str.keys():
    X_test[i].fillna(impute_value_str[i], inplace=True)
for i in apply_value_log:
    X_test['log_'+i]=np.log1p(X_test[i])
    

print(X_test.isna().sum())
X_test.sort_index(axis=1, inplace=True)
X_test_fitted=column_trans.transform(X_test)

# X_train_df=pd.DataFrame(X_train_fitted)
# X_test_df=pd.DataFrame(X_test_fitted)
# rfecv.fit(X_train_fitted,Y_train)
# rfecv.score(X_test_fitted,Y_test)

# 4.5 tranformation of dependent 
#histogram
import seaborn as sns
Y_train_Log1p = np.log1p(Y_train)
sns.displot(Y_train_Log1p)
sns.displot(Y_train)

Y_test_Log1p = np.log1p(Y_test)

### 5 modeling
# 5.1 modeling on train

import lightgbm as lgb

d_train = lgb.Dataset(X_train_fitted, label=Y_train_Log1p)
params = {}
params['learning_rate'] = 0.07
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'rmse'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 50
params['max_depth'] = 10

model = lgb.train(params, d_train, 100)

    
from sklearn.metrics import mean_squared_error, r2_score
  
Y_train_model=Y_train_Log1p
Y_test_model=Y_test_Log1p
  
#Train
print('Mean squared error Train: %.2f'
      % mean_squared_error(Y_train_model, model.predict(X_train_fitted)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination R2 Train: %.2f'
      % r2_score(Y_train_model, model.predict(X_train_fitted)))

#Test
print('Mean squared error Test: %.2f'
      % mean_squared_error(Y_test_model, model.predict(X_test_fitted, predict_disable_shape_check=True)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination R2 Test: %.2f'
      % r2_score(Y_test_model, model.predict(X_test_fitted, predict_disable_shape_check=True)))


X_test['predict']=np.expm1(model.predict(X_test_fitted))
final_df=X_test.merge(Y_test_model, on='Id')
final_df['diff']=final_df['predict']-final_df['SalePrice']

# visual 
plt.scatter(final_df['predict'],final_df['SalePrice'])
plt.show()
plt.clf()

# apply model to validation
df_val=pd.read_csv(full_path+"\\test.csv", index_col='Id')

print(df_val.isna().sum())
str_lst_flags=['Alley','Street','MiscFeature','FireplaceQu','GarageQual','Fence']
for i in str_lst_flags:
    df_val['f_'+str(i)]=df_val[i].apply(lambda x: create_flags(x))

df_val.drop(columns=drop_list,axis='columns', inplace=True)
for i in impute_value_num.keys():
    df_val[i].fillna(impute_value_num[i], inplace=True)
for i in impute_value_str.keys():
    df_val[i].fillna(impute_value_str[i], inplace=True)

impute_var_v4(df=df_val,var=str_var_nonan,perc_drop=1,style='mode')
impute_var_v4(df=df_val,var=num_var_nonan,perc_drop=1,style='median')

for i in apply_value_log:
    df_val['log_'+i]=np.log1p(df_val[i])

df_val['TotalSF'] = df_val['TotalBsmtSF'] + df_val['1stFlrSF'] + df_val['2ndFlrSF']
df_val['Total_Bathrooms'] = (df_val['FullBath'] + (0.5 * df_val['HalfBath']) + df_val['BsmtFullBath'] + (0.5 * df_val['BsmtHalfBath']))


print(df_val.isna().sum().sort_values())
df_val.sort_index(axis=1, inplace=True)
df_val_fitted=column_trans.transform(df_val)

df_val['SalePrice']= np.expm1(model.predict(df_val_fitted))

#output
out=df_val['SalePrice']
out.to_csv(path_or_buf=full_path+"\\result.csv",index=True)





# other things
# features_to_remove = ['address','attributes','business_id','categories','city','hours','is_open','latitude','longitude','name','neighborhood','postal_code','state','time']
# df.drop(labels=features_to_remove, axis=1, inplace=True)()

# sorted(list(zip(['average_review_length','average_review_age'],model.coef_)),key = lambda x: abs(x[1]),reverse=True)

# pd.DataFrame(list(zip(features.columns,features.describe().loc['mean'],features.describe().loc['min'],features.describe().loc['max'])),columns=['Feature','Mean','Min','Max'])



# var_list = ["CreditScore","Age","Tenure","Balance","NumOfProducts","EstimatedSalary","Exited"]
# sns.heatmap(churn_data[var_list].corr(), annot = True)
# plt.title("Correlation Matrix")
# plt.tight_layout()
# plt.show()
# plt.clf()
