# -*- coding: utf-8 -*-


import os
path="C:\\Users\\thoma\\Desktop\\VM share\\Python\\Kaggle\\helper"
os.chdir(path)
print(os.chdir(path))
%run helper.py

def import_file(full_path):
    import pandas as pd
    df=pd.read_csv(full_path+"\\train.csv")
    return df

full_path="C:\\Users\\thoma\\Desktop\\VM share\\Python\\files\\Kaggle\\titanic"
df=import_file(full_path)


# get_violinplot_for_target(df=df
#                 ,var=['Cabin_new'] 
#                 ,target='Survived')


#create new variables:
df['Cabin_new']=df['Cabin'].str[0:1]
plot_str_var(df,['Cabin_new'])

# sibsp	# of siblings / spouses aboard the Titanic	
# parch	# of parents / children aboard the Titanic
import numpy as np
## Create categorical variable for traveling alone
df['TravelAlone']=np.where((df["SibSp"]+df["Parch"])>0, 0, 1)
df['f_SibSp']=np.where((df["SibSp"])>0, 0, 1)
df['f_Parch']=np.where((df["Parch"])>0, 0, 1)
df['IsMinor']=np.where(df['Age']<=16, 1, 0)
df['Family']=df["SibSp"]+df["Parch"]

def fare_binning(Fare):
  if Fare <= 10.481249809265137:
    if Fare <= 7.133349895477295:
      return 1
    else:  # if Fare > 7.133349895477295
      return 2
  else:  # if Fare > 10.481249809265137
    if Fare <= 74.375:
      return 3
    else:  # if Fare > 74.375
      return 4
  
df['Fare_bin']=df['Fare'].apply(lambda x: fare_binning(x))

def age_binning(Age):
  if Age <= 40.25:
    if Age <= 17.5:
      return 1
    else:  # if Age > 17.5
      return 2
  else:  # if Age > 40.25
    if Age <= 58.5:
      return 3
    else:  # if Age > 58.5
      return 4

df['Age_bin']=df['Age'].apply(lambda x: age_binning(x))


independent=df.drop(columns='Survived')
dependent=df['Survived']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(independent, dependent, train_size=0.3, random_state=41, shuffle=False)

X_train.drop(columns=['Cabin'], inplace=True)
X_train.drop('SibSp', axis=1, inplace=True)
X_train.drop('Parch', axis=1, inplace=True)


all_var, num_var_nan, num_var_nonan, str_var_nan, str_var_nonan= pre_work(X_train)   

plot_num_var(X_train,num_var_nan)
plot_str_var(X_train,str_var_nan)

drop_list_num,impute_value_num=impute_var_v4(df=X_train,var=num_var_nan,perc_drop=0.25,style='median')
drop_list_str,impute_value_str=impute_var_v4(df=X_train,var=str_var_nan,perc_drop=1,style='nan')

drop_list_lowCor=merge_low_corr(df_ind=X_train, df_dep=Y_train, target='Survived', min_cor=0.05)
drop_list_max_perc_rep=same_value(df=X_train,var=all_var,max_perc_rep=0.95)


drop_list_woTarget=(drop_list_num+drop_list_str
                    #+drop_list_lowCor
                    +drop_list_max_perc_rep+['Name','Ticket','PassengerId'])
# drop_list_woTarget.remove('SibSp')
drop_list=(drop_list_woTarget)


X_train.drop(columns=drop_list_woTarget, inplace=True)
import pandas as pd
X_train=pd.get_dummies(X_train, columns=["Pclass","Embarked","Sex","Cabin_new","Age_bin","Fare_bin"])
X_train['Cabin_new_T']=0

X_train.sort_index(ascending=True, axis=1, inplace=True)

# Standardize values
# SC=Standardize_values(X_train)
# X_train = SC.transform(X_train)

#Normalie values
# TF=Normalize_values(X_train)
# X_train = TF.transform(X_train)


def apply_transformation(df, drop_list_woTarget):
    for i in impute_value_num.keys():
        df[i].fillna(impute_value_num[i], inplace=True)
    for i in impute_value_str.keys():
        df[i].fillna(impute_value_str[i], inplace=True)

    
    # impute_var_v4(df=df,var=num_var_nan,perc_drop=1,style='median')
    # impute_var_v4(df=df,var=str_var_nan,perc_drop=1,style='nan')
    impute_var_v4(df=df,var=num_var_nonan,perc_drop=1,style='mean')
    #df.dropna(axis=0, how='any', inplace=True)
    
    df.drop(columns=['Cabin'], inplace=True)
    df.drop('SibSp', axis=1, inplace=True)
    df.drop('Parch', axis=1, inplace=True)
    df.drop(columns=drop_list_woTarget, inplace=True)
    
    df=pd.get_dummies(df,columns=["Pclass","Embarked","Sex","Cabin_new","Age_bin","Fare_bin"])
    
    df.sort_index(ascending=True, axis=1, inplace=True)

    #df['Cabin_new_T']=0
    # df = SC.transform(df)
    #df=TF.transform(df)
    return df

X_test = apply_transformation(X_test, drop_list_woTarget)

#print(X_test.isna().sum())

# print(list(X_test.columns))
# print(list(X_train.columns))
# intersection(list(X_test.columns), list(X_train.columns))
# no_intersection(list(X_test.columns), list(X_train.columns))
# no_intersection(list(X_train.columns),list(X_test.columns))


from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

# cols = ["Age","Fare","TravelAlone","Pclass_1","Pclass_2","Embarked_C","Embarked_S","Sex_male","IsMinor"] 
# X = final_train[cols]
# y = final_train['Survived']
# # Build a logreg and compute the feature importances
model = LogisticRegression(max_iter=1000,solver='liblinear', random_state=41)
# create the RFE model and select 8 attributes
rfe = RFE(model, n_features_to_select=len(X_train.columns)) 
rfe = rfe.fit(X_train, Y_train)
# summarize the selection of the attributes
print("Optimal number of features : %d" % rfe.n_features_)
print('Selected features: %s' % list(X_train.columns[rfe.support_]))



#Recursive feature elimination
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

min_features_to_select = 1  # Minimum number of features to consider

rfecv = RFECV(estimator=model
              ,step=min_features_to_select
              ,cv=StratifiedKFold(10)
              ,scoring='accuracy')
rfecv.fit(X_train, Y_train)

print("Optimal number of features : %d" % rfecv.n_features_)
print('Selected features: %s' % list(X_train.columns[rfecv.support_]))

# Plot number of features VS. cross-validation scores
plt.figure(figsize=(10,6))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(min_features_to_select,
               len(rfecv.grid_scores_) + min_features_to_select),
         rfecv.grid_scores_)
plt.show()


var=list(X_train.columns[rfecv.support_])
#var_1=['Cabin_new_B', 'Cabin_new_D', 'Cabin_new_F', 'Cabin_new_G', 'Cabin_new_missing', 'Embarked_Q', 'Embarked_S', 'IsMinor', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 'f_SibSp']
#var_2=['Cabin_new_B', 'Cabin_new_D', 'Cabin_new_F', 'Cabin_new_G', 'Cabin_new_missing', 'Embarked_Q', 'Embarked_S', 'IsMinor', 'Pclass_3', 'Sex_female', 'Sex_male', 'TravelAlone', 'f_SibSp']
#var_3=['Age','Age_bin_1', 'Age_bin_2', 'Age_bin_3', 'Age_bin_4', 'Cabin_new_A', 'Cabin_new_B', 'Cabin_new_C', 'Cabin_new_D', 'Cabin_new_E', 'Cabin_new_F', 'Cabin_new_G', 'Cabin_new_T', 'Cabin_new_missing', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Embarked_missing', 'Family', 'Fare', 'Fare_bin_1', 'Fare_bin_2', 'Fare_bin_3', 'Fare_bin_4', 'IsMinor', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 'TravelAlone', 'f_Parch', 'f_SibSp']
#var=['Cabin_new_B', 'Cabin_new_D', 'Cabin_new_E', 'Cabin_new_F', 'Cabin_new_G', 'Cabin_new_missing', 'Embarked_Q', 'Fare_bin_1', 'Fare_bin_3', 'Fare_bin_4', 'IsMinor', 'Pclass_3', 'Sex_female', 'Sex_male', 'TravelAlone']


X_train=X_train[var]
X_test=X_test[var]

get_heatmap(X_train)

model.fit(X_train, Y_train)

# calculated_coefficients = model.coef_ 
# intercept = model.intercept_
model_param= dict(zip(np.array(X_train.columns),model.coef_[0]))
model_param['intercept']=model.intercept_[0]


## check performance:

from sklearn.metrics import accuracy_score, log_loss, auc, roc_curve, mean_squared_error, r2_score

Y_pred_train = model.predict(X_train)
Y_pred_proba_train = model.predict_proba(X_train)[:, 1]

Y_pred_test = model.predict(X_test)
Y_pred_proba_test = model.predict_proba(X_test)[:, 1]
[fpr, tpr, thr] = roc_curve(Y_test, Y_pred_proba_test)

print('Train/Test split results:')
print(model.__class__.__name__+" Train mean squared error is %.2f" % mean_squared_error(Y_train, Y_pred_train))
print(model.__class__.__name__+" Train R2 is %.2f" % r2_score(Y_train, Y_pred_train))
print(model.__class__.__name__+" Test mean squared error is %.2f" % mean_squared_error(Y_test, Y_pred_test))
print(model.__class__.__name__+" Test R2 is %.2f" % r2_score(Y_test, Y_pred_test))
print(model.__class__.__name__+" Test accuracy is %2.3f" % accuracy_score(Y_test, Y_pred_test))
print(model.__class__.__name__+" Test log loss is %2.3f" % log_loss(Y_test, Y_pred_proba_test))
print(model.__class__.__name__+" Test auc is %2.3f" % auc(fpr, tpr))



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


df_val=pd.read_csv(full_path+"\\test.csv")
X_val=df_val.copy(deep=True)
X_val['Cabin_new']=X_val['Cabin'].str[0:1]
#plot_str_var(X_val,['Cabin_new'])
X_val['Embarked_missing']=0
X_val['Cabin_new_T']=0
X_val['TravelAlone']=np.where((X_val["SibSp"]+X_val["Parch"])>0, 0, 1)
X_val['f_SibSp']=np.where((X_val["SibSp"])>0, 0, 1)
X_val['f_Parch']=np.where((X_val["Parch"])>0, 0, 1)
X_val['IsMinor']=np.where(X_val['Age']<=16, 1, 0)
X_val['Family']=X_val["SibSp"]+df["Parch"]
X_val['Fare_bin']=X_val['Fare'].apply(lambda x: fare_binning(x))
X_val['Age_bin']=X_val['Age'].apply(lambda x: age_binning(x))


X_val= apply_transformation(df=X_val, drop_list_woTarget=drop_list_woTarget)
#print(X_val.isna().sum())


# no_intersection(list(X_train.columns),list(X_val.columns))

df_val['Survived']= model.predict(X_val[var])

out=df_val[['PassengerId', 'Survived']]
out.to_csv(path_or_buf=full_path+"\\result.csv",index=False)

