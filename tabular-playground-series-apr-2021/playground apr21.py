### 1. import helper
import os
path="C:\\Users\\thoma\\Desktop\\VM share\\Python\\Kaggle\\helper"
os.chdir(path)
%run helper.py


### 2. source data
full_path="C:\\Users\\thoma\\Desktop\\VM share\\Python\\files\\Kaggle\\tabular-playground-series-apr-2021"

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

# 3.2 missing values
get_value_num_nan=get_var_value_counts(df_stacked,num_var_nan)
get_value_str_nan=get_var_value_counts(df_stacked,str_var_nan)
getMissingValues(df_stacked)

# 3.3 plot variables:
# plot_str_var(df=df_stacked[df_stacked['f_train']==1],var=str_var_nan)
plot_str_var(df=df_stacked[df_stacked['f_train']==1],var=['Embarked'])
plot_num_var(df=df_stacked[df_stacked['f_train']==1],var=num_var_nan)

# 3.4 corr
df_high_corr, high_corr_var=get_df_high_corr_target(df=df_stacked[df_stacked['f_train']==1],target='Survived',min_cor=0.5)
# get_scatter_for_target(df=df_stacked[df_stacked['f_train']==1],var=high_corr_var,target='Survived')

# 3.5 impute without data leakage
# drop_list_str,impute_value_str=impute_var_v4(df=df,var=str_var_nan,perc_drop=1,style='value',value='NA')


### 4 pred data
# 4.1 Build a Pipeline
# 4.1.1 Lib import
#general
import numpy as np
import pandas as pd

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
# ### add here iterative imputer
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer

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


# 4.1.2 pipeline steps
# 4.1.2.1 pre impute
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
              
        X["SibSp"].fillna(0)
        X["Parch"].fillna(0)
                
        return X

# 4.1.2.2 impute
impute = DataFrameMapper(
      [([item],SimpleImputer(strategy="constant",fill_value='NA')) for item in str_var_nan]
    + [([item],SimpleImputer(strategy="constant",fill_value='NA')) for item in str_var_nonan]
    + [([item],SimpleImputer(strategy='median')) for item in num_var_nan]
    + [([item],SimpleImputer(strategy='median')) for item in num_var_nonan]
    ,drop_cols=None
    ,default=None
    ,df_out =True
    )

# 4.1.2.3 post impute
class AddVariablesImputed(TransformerMixin):

    def __init__(self):
        return None
    
    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        import pandas as pd
        X['TravelAlone']=np.where((X["SibSp"]+X["Parch"])>0, 0, 1)
        X['f_SibSp']=np.where((X["SibSp"])>0, 1, 0)
        X['f_Parch']=np.where((X["Parch"])>0, 1, 0)
        X['IsMinor']=np.where(X['Age']<=16, 1, 0)
        
        X['Family_Size']=X["SibSp"]+X["Parch"]             
        lst=['Fare','Age']
        for i in lst:
            X[i+'_bin_vol'] = pd.qcut(x=X[i],q=10,labels=False)
            X[i+'_bin_int'] = pd.cut(x=X[i],bins=10,labels=False)
        
        def convert_Pclass(x):
            if x==3:
                return 1
            elif x==1:
                return 3
            else:
                return 2
        
        X['PclassConv']=X['Pclass'].apply(lambda x: convert_Pclass(x))
        X['Inter_PclassConv_Fare']=X['PclassConv']*X['Fare']
        return X

# 4.1.2.4 transformation
nominal=['Embarked','Sex','Pclass','f_Cabin','f_Ticket','Fare_bin_vol','Fare_bin_int','Age_bin_vol','Age_bin_int']
# categorial=[]
# category_array=[]
# column_to_cat=dict(zip(categorial,category_array))
skew_var=determine_skewed_var(df=df_stacked[df_stacked['f_train']==1]
                              ,num_var=num_var_nan+num_var_nonan
                              ,factor=0.5)
drop=['Cabin','Name','Ticket']

transformation = DataFrameMapper(
    [([item],OneHotEncoder(handle_unknown='ignore')) for item in nominal]
    # + [([col],OrdinalEncoder(categories = [cat])) for col, cat in column_to_cat.items()]
    + [([item],PowerTransformer()) for item in skew_var]
    # +[([item],None) for item in pass_through]
    ,default=None #None #False
    ,df_out=True
    , drop_cols=drop
)

# 4.1.2.4 post transformation
class AddVariablesTransformed(TransformerMixin):

    def __init__(self):
        return None
    
    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):

        return X
        

# 4.2 model selection
# 4.2.1 split dependent and independent
dependent=df_train['Survived']
independent=df_train.copy(deep=True)
independent.drop(columns='Survived',axis='columns', inplace=True)

# 4.2.2 apply pipeline
pipeline_pre = make_pipeline(
    AddVariablesNotImputed()
    ,(impute)
    ,AddVariablesImputed()
    ,(transformation)
    ,AddVariablesTransformed()
    ,(VarianceThreshold(0.1))
    ,(StandardScaler())
)

independent_trans=pipeline_pre.fit_transform(independent.copy())

# 4.2.3 select model
best_model4=pre_eval_models(
    type_model='classification',
    independent=independent_trans, 
    dependent=dependent,
    scoring='accuracy',
    cv=5)

# 4.3 split test train data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(independent,dependent,test_size=0.2, random_state=41, shuffle=True)

# 4.4 Hyperparameter tuning
# 4.4.1 General Pipeline wo model
pipeline_model = make_pipeline(
    AddVariablesNotImputed()
    ,(impute)
    ,AddVariablesImputed()
    ,(transformation)
    ,AddVariablesTransformed()
    ,(VarianceThreshold(0.1))
    ,(StandardScaler())
    #,RFE(estimator=SVR(kernel="linear"), step= 3, n_features_to_select=0.8)
)

# pipeline.get_params().keys()

# 4.4.2 import
#Model 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron   
#Stacking
from sklearn.ensemble import StackingClassifier as stack

# 4.4.3.1 logisticregression
params={'rfe__estimator': [SVR(kernel="linear")]
        ,'rfe__n_features_to_select': [0.5,0.8]
        ,'rfe__step':[3]
        ,'logisticregression__C': np.logspace(-4, 4, 5)
        ,'logisticregression__penalty' : ['l1', 'l2']
        ,'logisticregression__solver' : ['liblinear']
        }

pipeline = make_pipeline(
    RFE(estimator=SVR(kernel="linear"))
    ,LogisticRegression()
    )

# pipeline.get_params().keys()

trans_model =RandomizedSearchCV(pipeline
                          ,params
                          ,scoring='accuracy'
                          ,cv = 2
                          ,n_jobs=-1)
trans_model.fit(pipeline_model.fit_transform(X_train.copy()), Y_train.copy())

for param, score in zip(trans_model.cv_results_['params'], trans_model.cv_results_['mean_test_score']):
    print (param, score)
print(trans_model.best_params_)

# {'rfe__step': 3, 'rfe__n_features_to_select': 0.8, 'rfe__estimator': SVR(kernel='linear'), 'logisticregression__solver': 'liblinear', 'logisticregression__penalty': 'l2', 'logisticregression__C': 10000.0} 0.7718875
# {'rfe__step': 3, 'rfe__n_features_to_select': 0.8, 'rfe__estimator': SVR(kernel='linear'), 'logisticregression__solver': 'liblinear', 'logisticregression__penalty': 'l2', 'logisticregression__C': 100.0} 0.7719
# {'rfe__step': 3, 'rfe__n_features_to_select': 0.5, 'rfe__estimator': SVR(kernel='linear'), 'logisticregression__solver': 'liblinear', 'logisticregression__penalty': 'l2', 'logisticregression__C': 1.0} 0.7624124999999999
# {'rfe__step': 3, 'rfe__n_features_to_select': 0.8, 'rfe__estimator': SVR(kernel='linear'), 'logisticregression__solver': 'liblinear', 'logisticregression__penalty': 'l1', 'logisticregression__C': 0.0001} 0.7574624999999999
# {'rfe__step': 3, 'rfe__n_features_to_select': 0.5, 'rfe__estimator': SVR(kernel='linear'), 'logisticregression__solver': 'liblinear', 'logisticregression__penalty': 'l1', 'logisticregression__C': 0.01} 0.7614125
# {'rfe__step': 3, 'rfe__n_features_to_select': 0.5, 'rfe__estimator': SVR(kernel='linear'), 'logisticregression__solver': 'liblinear', 'logisticregression__penalty': 'l1', 'logisticregression__C': 100.0} 0.76235
# {'rfe__step': 3, 'rfe__n_features_to_select': 0.5, 'rfe__estimator': SVR(kernel='linear'), 'logisticregression__solver': 'liblinear', 'logisticregression__penalty': 'l2', 'logisticregression__C': 10000.0} 0.762325
# {'rfe__step': 3, 'rfe__n_features_to_select': 0.8, 'rfe__estimator': SVR(kernel='linear'), 'logisticregression__solver': 'liblinear', 'logisticregression__penalty': 'l2', 'logisticregression__C': 0.0001} 0.7676625
# {'rfe__step': 3, 'rfe__n_features_to_select': 0.8, 'rfe__estimator': SVR(kernel='linear'), 'logisticregression__solver': 'liblinear', 'logisticregression__penalty': 'l2', 'logisticregression__C': 0.01} 0.7702249999999999
# {'rfe__step': 3, 'rfe__n_features_to_select': 0.8, 'rfe__estimator': SVR(kernel='linear'), 'logisticregression__solver': 'liblinear', 'logisticregression__penalty': 'l1', 'logisticregression__C': 1.0} 0.7717125
# {'rfe__step': 3, 'rfe__n_features_to_select': 0.8, 'rfe__estimator': SVR(kernel='linear'), 'logisticregression__solver': 'liblinear', 'logisticregression__penalty': 'l2', 'logisticregression__C': 100.0}

# evaluate_model(model_type='classification'
#                 ,model=trans_model
#                 ,X=X_test.copy()
#                 ,y_true=Y_test.copy()
#                 )

# 4.4.3.2 xgbclassifier
params={
        'xgbclassifier__objective': ['reg:gamma', 'reg:squarederror', 'reg:squaredlogerror']
        ,'xgbclassifier__booster': ['gbtree','gblinear','dart']
        ,'xgbclassifier__learning_rate': [0.01,0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        ,'xgbclassifier__n_estimators': [100,200,500,1000,2000,2500,5000]
        ,'xgbclassifier__max_depth': [3,7,11,12,13,15,30]
        # ,xgbregressor__colsample_bytree
        # ,xgbregressor__gamma
        # ,xgbregressor__reg_alpha
        # ,xgbregressor__reg_alpha
        # ,xgbregressor__subsample
        }

pipeline = make_pipeline(
    XGBClassifier()
    )

# pipeline.get_params().keys()

trans_model =RandomizedSearchCV(pipeline
                          ,params
                          ,scoring='accuracy'
                          ,cv = 2
                          ,n_jobs=-1)
trans_model.fit(pipeline_model.fit_transform(X_train.copy()), Y_train.copy())

for param, score in zip(trans_model.cv_results_['params'], trans_model.cv_results_['mean_test_score']):
    print (param, score)
print(trans_model.best_params_)

# {'xgbclassifier__objective': 'reg:squarederror', 'xgbclassifier__n_estimators': 2000, 'xgbclassifier__max_depth': 30, 'xgbclassifier__learning_rate': 0.2, 'xgbclassifier__booster': 'gblinear'} 0.7719875
# {'xgbclassifier__objective': 'reg:squarederror', 'xgbclassifier__n_estimators': 2500, 'xgbclassifier__max_depth': 12, 'xgbclassifier__learning_rate': 0.15, 'xgbclassifier__booster': 'gblinear'} 0.771975
# {'xgbclassifier__objective': 'reg:squarederror', 'xgbclassifier__n_estimators': 1000, 'xgbclassifier__max_depth': 12, 'xgbclassifier__learning_rate': 0.25, 'xgbclassifier__booster': 'gblinear'} 0.7716875000000001

# evaluate_model(model_type='classification'
#                 ,model=trans_model
#                 ,X=X_test.copy()
#                 ,y_true=Y_test.copy()
#                 )

# 4.4.3.3 gradientboostingclassifier
params ={'gradientboostingclassifier__learning_rate':[0.1,0.01]
         , 'gradientboostingclassifier__n_estimators':[100,1500]
         , 'gradientboostingclassifier__max_depth':[3,4]
         , 'gradientboostingclassifier__min_samples_split':[2,40]
         , 'gradientboostingclassifier__min_samples_leaf':[1,7]
         , 'gradientboostingclassifier__subsample':[1, 0.95]
         , 'gradientboostingclassifier__max_features':['sqrt']
    }

pipeline = make_pipeline(
    GradientBoostingClassifier()
    )

pipeline.get_params().keys()

trans_model =RandomizedSearchCV(pipeline
                          ,params
                          ,scoring='accuracy'
                          ,cv = 2
                          ,n_jobs=-1)
trans_model.fit(pipeline_model.fit_transform(X_train.copy()), Y_train.copy())

for param, score in zip(trans_model.cv_results_['params'], trans_model.cv_results_['mean_test_score']):
    print (param, score)
print(trans_model.best_params_)

# {'gradientboostingclassifier__subsample': 1, 'gradientboostingclassifier__n_estimators': 1500, 'gradientboostingclassifier__min_samples_split': 2, 'gradientboostingclassifier__min_samples_leaf': 7, 'gradientboostingclassifier__max_features': 'sqrt', 'gradientboostingclassifier__max_depth': 4, 'gradientboostingclassifier__learning_rate': 0.1} 0.7716000000000001
# {'gradientboostingclassifier__subsample': 1, 'gradientboostingclassifier__n_estimators': 100, 'gradientboostingclassifier__min_samples_split': 40, 'gradientboostingclassifier__min_samples_leaf': 1, 'gradientboostingclassifier__max_features': 'sqrt', 'gradientboostingclassifier__max_depth': 4, 'gradientboostingclassifier__learning_rate': 0.01} 0.768875
# {'gradientboostingclassifier__subsample': 0.95, 'gradientboostingclassifier__n_estimators': 1500, 'gradientboostingclassifier__min_samples_split': 40, 'gradientboostingclassifier__min_samples_leaf': 1, 'gradientboostingclassifier__max_features': 'sqrt', 'gradientboostingclassifier__max_depth': 4, 'gradientboostingclassifier__learning_rate': 0.01} 0.7761375
# {'gradientboostingclassifier__subsample': 1, 'gradientboostingclassifier__n_estimators': 1500, 'gradientboostingclassifier__min_samples_split': 40, 'gradientboostingclassifier__min_samples_leaf': 7, 'gradientboostingclassifier__max_features': 'sqrt', 'gradientboostingclassifier__max_depth': 4, 'gradientboostingclassifier__learning_rate': 0.1} 0.7723500000000001
# {'gradientboostingclassifier__subsample': 1, 'gradientboostingclassifier__n_estimators': 100, 'gradientboostingclassifier__min_samples_split': 40, 'gradientboostingclassifier__min_samples_leaf': 1, 'gradientboostingclassifier__max_features': 'sqrt', 'gradientboostingclassifier__max_depth': 4, 'gradientboostingclassifier__learning_rate': 0.1} 0.7761875
# {'gradientboostingclassifier__subsample': 1, 'gradientboostingclassifier__n_estimators': 100, 'gradientboostingclassifier__min_samples_split': 2, 'gradientboostingclassifier__min_samples_leaf': 7, 'gradientboostingclassifier__max_features': 'sqrt', 'gradientboostingclassifier__max_depth': 4, 'gradientboostingclassifier__learning_rate': 0.1} 0.7764375
# {'gradientboostingclassifier__subsample': 0.95, 'gradientboostingclassifier__n_estimators': 100, 'gradientboostingclassifier__min_samples_split': 2, 'gradientboostingclassifier__min_samples_leaf': 1, 'gradientboostingclassifier__max_features': 'sqrt', 'gradientboostingclassifier__max_depth': 3, 'gradientboostingclassifier__learning_rate': 0.1} 0.77605
# {'gradientboostingclassifier__subsample': 0.95, 'gradientboostingclassifier__n_estimators': 1500, 'gradientboostingclassifier__min_samples_split': 40, 'gradientboostingclassifier__min_samples_leaf': 1, 'gradientboostingclassifier__max_features': 'sqrt', 'gradientboostingclassifier__max_depth': 4, 'gradientboostingclassifier__learning_rate': 0.1} 0.7709
# {'gradientboostingclassifier__subsample': 0.95, 'gradientboostingclassifier__n_estimators': 1500, 'gradientboostingclassifier__min_samples_split': 40, 'gradientboostingclassifier__min_samples_leaf': 1, 'gradientboostingclassifier__max_features': 'sqrt', 'gradientboostingclassifier__max_depth': 3, 'gradientboostingclassifier__learning_rate': 0.01} 0.7763
# {'gradientboostingclassifier__subsample': 1, 'gradientboostingclassifier__n_estimators': 1500, 'gradientboostingclassifier__min_samples_split': 2, 'gradientboostingclassifier__min_samples_leaf': 7, 'gradientboostingclassifier__max_features': 'sqrt', 'gradientboostingclassifier__max_depth': 4, 'gradientboostingclassifier__learning_rate': 0.01} 0.7761375
# {'gradientboostingclassifier__subsample': 1, 'gradientboostingclassifier__n_estimators': 100, 'gradientboostingclassifier__min_samples_split': 2, 'gradientboostingclassifier__min_samples_leaf': 7, 'gradientboostingclassifier__max_features': 'sqrt', 'gradientboostingclassifier__max_depth': 4, 'gradientboostingclassifier__learning_rate': 0.1}

# evaluate_model(model_type='classification'
#                 ,model=trans_model
#                 ,X=X_test.copy()
#                 ,y_true=Y_test.copy()
#                 )

# 4.4.3.4 lgbmclassifier
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
params ={'lgbmclassifier__num_leaves': sp_randint(6, 50), 
              'lgbmclassifier__min_child_samples': sp_randint(100, 500), 
              'lgbmclassifier__min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
              'lgbmclassifier__subsample': sp_uniform(loc=0.2, scale=0.8), 
              'lgbmclassifier__colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
              'lgbmclassifier__reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
              'lgbmclassifier__reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}

pipeline = make_pipeline(
    LGBMClassifier()
    )

# pipeline.get_params().keys()

trans_model =RandomizedSearchCV(pipeline
                          ,params
                          ,scoring='accuracy'
                          ,cv = 2
                          ,n_jobs=-1)

trans_model.fit(pipeline_model.fit_transform(X_train.copy()), Y_train.copy())

for param, score in zip(trans_model.cv_results_['params'], trans_model.cv_results_['mean_test_score']):
    print (param, score)
print(trans_model.best_params_)

# {'lgbmclassifier__colsample_bytree': 0.7276651575989284, 'lgbmclassifier__min_child_samples': 425, 'lgbmclassifier__min_child_weight': 10.0, 'lgbmclassifier__num_leaves': 12, 'lgbmclassifier__reg_alpha': 2, 'lgbmclassifier__reg_lambda': 1, 'lgbmclassifier__subsample': 0.4919101743705352} 0.7733
# {'lgbmclassifier__colsample_bytree': 0.9129433179732165, 'lgbmclassifier__min_child_samples': 454, 'lgbmclassifier__min_child_weight': 1e-05, 'lgbmclassifier__num_leaves': 26, 'lgbmclassifier__reg_alpha': 5, 'lgbmclassifier__reg_lambda': 5, 'lgbmclassifier__subsample': 0.5640600110185826} 0.7725
# {'lgbmclassifier__colsample_bytree': 0.6910911979680983, 'lgbmclassifier__min_child_samples': 418, 'lgbmclassifier__min_child_weight': 1000.0, 'lgbmclassifier__num_leaves': 43, 'lgbmclassifier__reg_alpha': 2, 'lgbmclassifier__reg_lambda': 10, 'lgbmclassifier__subsample': 0.5483483081741058} 0.7709874999999999
# {'lgbmclassifier__colsample_bytree': 0.7955122947510889, 'lgbmclassifier__min_child_samples': 305, 'lgbmclassifier__min_child_weight': 10.0, 'lgbmclassifier__num_leaves': 46, 'lgbmclassifier__reg_alpha': 7, 'lgbmclassifier__reg_lambda': 0.1, 'lgbmclassifier__subsample': 0.4021678156485233} 0.7723249999999999
# {'lgbmclassifier__colsample_bytree': 0.7776678287265193, 'lgbmclassifier__min_child_samples': 266, 'lgbmclassifier__min_child_weight': 0.001, 'lgbmclassifier__num_leaves': 7, 'lgbmclassifier__reg_alpha': 0, 'lgbmclassifier__reg_lambda': 0, 'lgbmclassifier__subsample': 0.27524861042130344} 0.7731874999999999
# {'lgbmclassifier__colsample_bytree': 0.48959256383430316, 'lgbmclassifier__min_child_samples': 312, 'lgbmclassifier__min_child_weight': 0.1, 'lgbmclassifier__num_leaves': 26, 'lgbmclassifier__reg_alpha': 50, 'lgbmclassifier__reg_lambda': 10, 'lgbmclassifier__subsample': 0.5726945920597315} 0.7714875
# {'lgbmclassifier__colsample_bytree': 0.9380505405191708, 'lgbmclassifier__min_child_samples': 435, 'lgbmclassifier__min_child_weight': 0.1, 'lgbmclassifier__num_leaves': 12, 'lgbmclassifier__reg_alpha': 7, 'lgbmclassifier__reg_lambda': 20, 'lgbmclassifier__subsample': 0.4840153213470245} 0.77295
# {'lgbmclassifier__colsample_bytree': 0.707486250591146, 'lgbmclassifier__min_child_samples': 159, 'lgbmclassifier__min_child_weight': 0.01, 'lgbmclassifier__num_leaves': 25, 'lgbmclassifier__reg_alpha': 7, 'lgbmclassifier__reg_lambda': 50, 'lgbmclassifier__subsample': 0.43288586128379747} 0.7730375
# {'lgbmclassifier__colsample_bytree': 0.8931953395239276, 'lgbmclassifier__min_child_samples': 405, 'lgbmclassifier__min_child_weight': 10000.0, 'lgbmclassifier__num_leaves': 36, 'lgbmclassifier__reg_alpha': 50, 'lgbmclassifier__reg_lambda': 5, 'lgbmclassifier__subsample': 0.49819479703627945} 0.5723625
# {'lgbmclassifier__colsample_bytree': 0.9989342306444497, 'lgbmclassifier__min_child_samples': 299, 'lgbmclassifier__min_child_weight': 0.001, 'lgbmclassifier__num_leaves': 38, 'lgbmclassifier__reg_alpha': 50, 'lgbmclassifier__reg_lambda': 20, 'lgbmclassifier__subsample': 0.6583283606681816} 0.7714749999999999
# {'lgbmclassifier__colsample_bytree': 0.7276651575989284, 'lgbmclassifier__min_child_samples': 425, 'lgbmclassifier__min_child_weight': 10.0, 'lgbmclassifier__num_leaves': 12, 'lgbmclassifier__reg_alpha': 2, 'lgbmclassifier__reg_lambda': 1, 'lgbmclassifier__subsample': 0.4919101743705352}

# evaluate_model(model_type='classification'
#                 ,model=trans_model
#                 ,X=X_test.copy()
#                 ,y_true=Y_test.copy()
#                 )

# 4.4.3.4 CatBoostClassifier
params = {'depth':[3,1,2,6,4,5,7,8,9,10]
          , 'iterations':[250,100,500,1000]
          # , 'learning_rate':[0.03,0.001,0.01,0.1,0.2,0.3]
          , 'l2_leaf_reg':[3,1,5,10,100]
          , 'border_count':[32,5,10,20,50,100,200]
          # ,'ctr_border_count':[50,5,10,20,100,200]
          # ,'thread_count':4
          }

pipeline = make_pipeline(
    CatBoostClassifier(**params)
    )

# pipeline.get_params().keys()

trans_model =RandomizedSearchCV(pipeline
                          ,params
                          ,scoring='accuracy'
                          ,cv = 2
                          ,n_jobs=-1)
trans_model.fit(pipeline_model.fit_transform(X_train.copy()), Y_train.copy())

for param, score in zip(trans_model.cv_results_['params'], trans_model.cv_results_['mean_test_score']):
    print (param, score)
print(trans_model.best_params_)

# evaluate_model(model_type='classification'
#                 ,model=trans_model
#                 ,X=X_test.copy()
#                 ,y_true=Y_test.copy()
#                 )

# 4.4.3.5 RandomForestClassifier
#     # {'classifier' : [RandomForestClassifier()],
#     # 'classifier__n_estimators' : list(range(10,101,10)),
#     # 'classifier__max_features' : list(range(6,32,5))}


### final model
from sklearn.ensemble import StackingClassifier as stack
estimators = [#('CAT',CatBoostClassifier())
              ('LGB',LGBMClassifier(is_unbalance = True))
              ,('XGB',XGBClassifier(use_label_encoder=False))
              ,('GBC',GradientBoostingClassifier())
              ,('MLP',MLPClassifier())
              ,('Log',LogisticRegression())
              ,('PER',Perceptron())
              ]

pipeline = make_pipeline(
    AddVariablesNotImputed()
    ,(impute)
    ,AddVariablesImputed()
    ,(transformation)
    ,RFE(estimator=DecisionTreeClassifier(), step= 3, n_features_to_select=0.8)
    ,stack(estimators=estimators, final_estimator=CatBoostClassifier()
        ,cv=2, n_jobs=-1, passthrough=True, verbose=1)
)

trans_model=pipeline.fit(X_train.copy(), Y_train.copy())

evaluate_model(model_type='classification'
                ,model=trans_model
                ,X=X_test.copy()
                ,y_true=Y_test.copy()
                )

#               precision    recall  f1-score   support

#            0       0.80      0.81      0.81     11437
#            1       0.75      0.72      0.73      8563

#     accuracy                           0.78     20000
#    macro avg       0.77      0.77      0.77     20000
# weighted avg       0.77      0.78      0.77     20000

# 0.7335467805051583
# 7.760947693324416
# [[9320 2117]
#  [2377 6186]]
# 0.7753

#               precision    recall  f1-score   support

#            0       0.80      0.82      0.81     11437
#            1       0.75      0.72      0.73      8563

#     accuracy                           0.78     20000
#    macro avg       0.77      0.77      0.77     20000
# weighted avg       0.77      0.78      0.78     20000

# 0.7332382310984308
# 7.750585100889027
# [[9344 2093]
#  [2395 6168]]
# 0.7756

# ['is_unbalance'] = True
#               precision    recall  f1-score   support

#            0       0.80      0.82      0.81     11437
#            1       0.75      0.72      0.73      8563

#     accuracy                           0.78     20000
#    macro avg       0.77      0.77      0.77     20000
# weighted avg       0.78      0.78      0.78     20000

# 0.7344363766218306
# 7.705683732058728
# [[9368 2069]
#  [2393 6170]]
# 0.7769

#RFE(estimator=DecisionTreeClassifier(), step= 3, n_features_to_select=0.8)
#               precision    recall  f1-score   support

#            0       0.80      0.81      0.80     11437
#            1       0.74      0.72      0.73      8563

#     accuracy                           0.77     20000
#    macro avg       0.77      0.77      0.77     20000
# weighted avg       0.77      0.77      0.77     20000

# 0.7335501034584688
# 7.783399137357125
# [[9289 2148]
#  [2359 6204]]
# 0.77465


# 5. Apply model
# apply model to validation
df_test=pd.read_csv(full_path+"\\test.csv",index_col='PassengerId')

df_test['Survived']=trans_model.predict(df_test.copy())

#output
out=df_test['Survived']
out.to_csv(path_or_buf=full_path+"\\result.csv",index=True)

