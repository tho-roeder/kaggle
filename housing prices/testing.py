# -*- coding: utf-8 -*-

# one hot vs. get_dummies
# predict value within data set to better impute
# count different values for string

#Exterior1st: Exterior covering on house: AsbShng	Asbestos Shingles
#Exterior2nd: Exterior covering on house (if more than one material): AsbShng	Asbestos Shingles

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


### 3. analyse data
# 3.1 create overview of data set:
#getGeneralInformation(df)
all_var, num_var_nan, num_var_nonan, str_var_nan, str_var_nonan= pre_work(df)   

# 3.2 string variables:
plot_str_var(df=df,var=str_var_nan)
plot_str_var(df=df,var=['Street'])

# Variable treement String:  
str_lst_flags=['Alley','Street','MiscFeature','FireplaceQu','GarageQual','Fence']
for i in str_lst_flags:
    df['f_'+str(i)]=df[i].apply(lambda x: create_flags(x))

# for i in ['BsmtFinType1','BsmtFinType2']:
#     df[i].replace(['GLQ','ALQ','BLQ','Rec','LwQ','Unf',np.nan],[6,5,4,3,2,1,0],inplace=True)  

# # quality scale
# def convert_scale(x):
#     if x=="Ex":
#         return 5 
#     elif x=="Gd":
#         return 4 
#     elif x=="TA":
#         return 3
#     elif x=="Fa":
#         return 2 
#     elif x=="Po":
#         return 1
#     elif pd.isnull(x):
#         return 0


# str_lst_scale=['BsmtQual','BsmtCond','FireplaceQu','GarageQual','GarageCond','PoolQC'
#                ,'ExterQual','ExterCond','HeatingQC','KitchenQual']

# for i in str_lst_scale:
#     df['f_'+str(i)]=df[i].apply(lambda x: convert_scale(x))

# mode or missing (imputation after test/train split)
str_lst_mode=['MasVnrType','BsmtExposure','BsmtFinType1','BsmtFinType2','Electrical','GarageType','GarageFinish','Fence'] 

# 3.3 num variables:
plot_num_var(df=df,var=num_var_nan)
# median for num_var_nan

# 3.4 corr
df_high_corr, high_corr_var=get_df_high_corr_target(df=df,target='SalePrice',min_cor=0.5)
get_scatter_for_target(df=df,var=high_corr_var,target='SalePrice')


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

# test_001=X_train[X_train['LotFrontage'].isna() ==True]
# test_001_df_high_corr, test_001_high_corr_var=get_df_high_corr_target(df=X_train,target='LotFrontage',min_cor=0.9)
# get_scatter_for_target(df=df,var=test_001_high_corr_var,target='LotFrontage')
# test_002=X_train[X_train['LotFrontage'].isna() ==True][['LotArea','LotFrontage']]
# test_003=X_train[['LotArea','LotFrontage']]

# 4.3.2 impute string
drop_list_str,impute_value_str=impute_var_v4(df=X_train,var=str_var_nan,perc_drop=1,style='value',value='NA')
#drop_list_str,impute_value_str=impute_var_v4(df=X_train,var=str_lst_mode,perc_drop=1,style='mode')
drop_list_num,impute_value_num=impute_var_v4(df=X_train,var=num_var_nan,perc_drop=0.25,style='median')

# 4.3.3 dropping variables
drop_list_lowCor=merge_low_corr(df_ind=X_train, df_dep=Y_train, target='SalePrice', min_cor=0.05)
get_scatter_for_target(df=df,var=drop_list_lowCor,target='SalePrice')

all_var.remove('SalePrice')
drop_list_max_perc_rep=same_value(df=X_train,var=all_var,max_perc_rep=0.95)
get_violinplot_for_target(df=df,var=drop_list_max_perc_rep,target='SalePrice')

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

var_cat_value=get_var_value_counts(df=df,var=str_var_nan+str_var_nonan)

nominal=['MSSubClass','MSZoning','Street', 'Alley','LandContour','Utilities','LotConfig','Neighborhood',
         'Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st',
         'Exterior2nd','MasVnrType','Foundation','Heating','CentralAir','Electrical','GarageType',
         'PavedDrive','Fence','MiscFeature','SaleType','SaleCondition']
categorial=['LotShape','LandSlope','ExterQual','ExterCond','BsmtCond','BsmtExposure','BsmtFinType1',
            'BsmtFinType2','HeatingQC','KitchenQual','Functional','FireplaceQu','GarageFinish',
            'GarageQual','GarageCond','PoolQC']
category_array=[
    ['Reg','IR1', 'IR2', 'IR3'],
    ['Gtl','Mod','Sev'],
    ['Ex','Gd','TA','Fa','Po'],
    ['Ex','Gd','TA','Fa','Po'],
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

test_001 = X_train[nominal+categorial]

from sklearn.preprocessing import OrdinalEncoder

oe=OrdinalEncoder(categories=category_array)

test_001[categorial]=oe.fit_transform(test_001[categorial])


#sklearn.preprocessing.OneHotEncoder
#https://towardsdatascience.com/one-hot-encoding-multicollinearity-and-the-dummy-variable-trap-b5840be3c41a
#X_train=pd.get_dummies(X_train)
# OneHotEncoder is prefered as it creates an object that can be applied to train & validation df
from  sklearn.preprocessing import OneHotEncoder
oneHot_encoder = OneHotEncoder(sparse=True)
other=oneHot_encoder.fit_transform(test_001[nominal])


oneHot_encoder.categories_
from sklearn.compose import make_column_transformer

column_trans = make_column_transformer(
    (OneHotEncoder(), nominal),
    (OrdinalEncoder(categories=category_array), categorial),
    remainder='passthrough')

X_train=column_trans.fit_transform(X_train)
impute_var_v4(df=X_test,var=str_var_nan,perc_drop=1,style='value',value='NA')
X_test=column_trans.transform(X_test)

# from sklearn.pipeline import make_pipeline
# pipe = make_pipeline(column_trans, logreg)
# cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean()
1
# 4.4 apply transformation on X_test
print(X_test[str_var_nan + str_var_nonan].isna().sum())
print(X_test[num_var_nan + num_var_nonan].isna().sum())

# 4.4.1
impute_var_v4(df=X_test,var=str_var_nan,perc_drop=1,style='nan')
impute_var_v4(df=X_test,var=num_var_nan,perc_drop=0.25,style='median')
#X_test=pd.get_dummies(X_test)


X_test=oneHot_encoder.transform(X_test)
X_test.drop(columns=drop_list,inplace=True)


### 5 modeling
# 5.1 modeling on train
from sklearn.linear_model import LinearRegression
model=LinearRegression()

from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

min_features_to_select = 1  # Minimum number of features to consider

rfecv = RFECV(estimator=model
              ,step=min_features_to_select
              ,cv=StratifiedKFold(5)
              ,verbose=1
              # ,scoring='accuracy'
              )
rfecv.fit(X_train, Y_train)

print("Optimal number of features : %d" % rfecv.n_features_)
print('Selected features: %s' % list(X_train.columns[rfecv.support_]))

var_selected=list(X_train.columns[rfecv.support_])
# var_selected=['OverallQual', 'FullBath', 'f_PoolQC', 'Utilities_AllPub', 'LotConfig_CulDSac', 
#               'Neighborhood_ClearCr', 'Neighborhood_Crawfor', 'Neighborhood_NoRidge', 
#               'Neighborhood_NridgHt', 'Neighborhood_StoneBr', 'Condition2_PosA', 'Condition2_PosN', 
#               'BldgType_Twnhs', 'BldgType_TwnhsE', 'HouseStyle_2.5Unf', 'RoofMatl_ClyTile', 'RoofMatl_WdShake', 
#               'RoofMatl_WdShngl', 'Exterior1st_CemntBd', 'Exterior1st_ImStucc', 'Exterior1st_WdShing', 
#               'Exterior2nd_CmentBd', 'Exterior2nd_Stone', 'Foundation_Wood', 'BsmtQual_Ex', 'BsmtExposure_Gd', 
#               'HeatingQC_Po', 'Electrical_FuseP', 'KitchenQual_Ex', 'Functional_Maj2', 'Functional_Sev', 
#               'PoolQC_Fa', 'MiscFeature_Othr', 'SaleType_Con', 'SaleType_New']
model_param= dict(zip(var_selected[:-1],rfecv.estimator_.coef_))
model_param['intercept']=rfecv.estimator_.intercept_
X_train=X_train[var_selected]
X_test=X_test[var_selected]






# assessing result
from sklearn.metrics import mean_squared_error, r2_score

#Train
print("R^2 Train score:")
print(rfecv.estimator_.score(X_train, Y_train))
print('Mean squared error: %.2f'
      % mean_squared_error(Y_train, rfecv.estimator_.predict(X_train)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(Y_train, rfecv.estimator_.predict(X_train)))






#Test
print("R^2 Test score:")
print(reg.score(X_test, Y_test))

print('Mean squared error: %.2f'
      % mean_squared_error(Y_test, reg.predict(X_test)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(Y_test, reg.predict(X_test)))


X_test['predict']=reg.predict(X_test)
final_df=X_test.merge(Y_test, on='Id')
final_df['diff']=final_df['predict']-final_df['SalePrice']



#from sklearn.metrics import classification_report
#print(classification_report(Y_test, reg.predict(X_test)))

# visual 
# plt.scatter(final_df['predict'],final_df['SalePrice'])
# plt.show()
# plt.clf()

# # apply model to validation
# #df=pd.read_csv(os.getcwd()+path+"\\test.csv")
# df=pd.read_csv(full_path+"\\test.csv")

# test_independent_input=df[no_issue]
# test_issue_cnt=test_independent_input.isnull().sum()
# # independent_miss=independent.fillna('missing')
# test_independent_dum=pd.get_dummies(test_independent_input)

# df['SalePrice']= reg.predict(test_independent_dum)

#output
# out=df[['Id', 'SalePrice']]
# out.to_csv(path_or_buf=os.getcwd()+path+"\\result.csv",index=False)





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
