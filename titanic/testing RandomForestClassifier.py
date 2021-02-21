# -*- coding: utf-8 -*-

#Srate: Titanic Modeling: NBayes + DTree - 0.8253
#https://www.kaggle.com/pavlofesenko/titanic-extended
#current best version

import os
path="C:\\Users\\thoma\\Desktop\\VM share\\Python\\Kaggle\\helper"
os.chdir(path)
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
df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
title_names = (df['Title'].value_counts() < 10)
df['Title'] = df['Title'].apply(lambda x: 'missing' if title_names.loc[x] == True else x)

# was derived basis training data - no data leakage

def fare_binning(Fare):
  if Fare <= 10.481249809265137:
    if Fare <= 7.133349895477295:
      return 1
    else:  # if Fare > 7.133349895477295
      return 2
  else:  # if Fare > 10.481249809265137
    if Fare <= 52.277099609375:
      return 3
    else:  # if Fare > 52.277099609375
      return 4 
  
df['Fare_bin']=df['Fare'].apply(lambda x: fare_binning(x))

def age_binning(Age):
  if Age <= 6.0:
    if Age <= 2.5:
      return 1
    else:  # if Age > 2.5
      return 2
  else:  # if Age > 6.0
    if Age <= 63.5:
      return 3
    else:  # if Age > 63.5
      return 4
  
df['Age_bin']=df['Age'].apply(lambda x: age_binning(x))

independent=df.drop(columns='Survived')
dependent=df['Survived']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(independent, dependent, train_size=0.7, random_state=41, shuffle=False)

X_train.drop(columns=['Cabin'], inplace=True)
# X_train.drop('SibSp', axis=1, inplace=True)
# X_train.drop('Parch', axis=1, inplace=True)


all_var, num_var_nan, num_var_nonan, str_var_nan, str_var_nonan= pre_work(X_train)   

plot_num_var(X_train,num_var_nan)
plot_str_var(X_train,str_var_nan)

drop_list_num,impute_value_num=impute_var_v4(df=X_train,var=num_var_nan,perc_drop=0.25,style='median')
drop_list_str,impute_value_str=impute_var_v4(df=X_train,var=str_var_nan,perc_drop=1,style='nan')

drop_list_lowCor=merge_low_corr(df_ind=X_train, df_dep=Y_train, target='Survived', min_cor=0.05)
drop_list_max_perc_rep=same_value(df=X_train,var=all_var,max_perc_rep=0.95)

# tree, pic_fare= get_tree_pic(df_X=X_train, df_Y=Y_train, var=['Fare'])
# tree_to_code(tree=tree, feature_names=['Fare'])
# pic_fare

# tree, pic_age= get_tree_pic(df_X=X_train, df_Y=Y_train, var=['Age'])
# tree_to_code(tree=tree, feature_names=['Age'])
# pic_age

#get_bins(df=df,var=['Fare'],nbr_bins=3)


drop_list_woTarget=(drop_list_num+drop_list_str
                    #+drop_list_lowCor
                    +drop_list_max_perc_rep+['Name','Ticket','PassengerId'])
# drop_list_woTarget.remove('SibSp')
drop_list=(drop_list_woTarget)


X_train.drop(columns=drop_list_woTarget, inplace=True)
import pandas as pd
X_train=X_train[["Pclass", "Sex", "SibSp", "Parch"]]
X_train = pd.get_dummies(X_train)

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

    impute_var_v4(df=df,var=num_var_nonan,perc_drop=1,style='mean')

    df.drop(columns=['Cabin'], inplace=True)

    df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    df['Title'] = df['Title'].apply(lambda x: 'missing' if title_names.loc[x] == True else x)
    df.drop(columns=drop_list_woTarget, inplace=True)
    df=df[["Pclass", "Sex", "SibSp", "Parch"]]
    df = pd.get_dummies(df)
    
    df.sort_index(ascending=True, axis=1, inplace=True)

    return df

X_test = apply_transformation(X_test, drop_list_woTarget)


#print(X_test.isna().sum())

# print(list(X_test.columns))
# print(list(X_train.columns))
# intersection(list(X_test.columns), list(X_train.columns))
# no_intersection(list(X_test.columns), list(X_train.columns))
# no_intersection(list(X_train.columns),list(X_test.columns))


#################


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=2)
model.fit(X_train, Y_train)
predictions = model.predict(X_test)


#var=['Age', 'Cabin_new_missing', 'Family', 'Fare', 'Pclass_3', 'Sex_female', 'Sex_male', 'Title_Mr']

var=list(X_train.columns)
X_train=X_train[var]
X_test=X_test[var]

get_heatmap(X_train)

model.fit(X_train, Y_train)
print(model.score(X_test, Y_test))
print(model.feature_importances_)

from sklearn.metrics import accuracy_score, log_loss, auc, roc_curve, mean_squared_error, r2_score

Y_pred_train = model.predict(X_train)
Y_pred_proba_train = model.predict_proba(X_train)[:, 1]

Y_pred_test = model.predict(X_test)
Y_pred_proba_test = model.predict_proba(X_test)[:, 1]
[fpr, tpr, thr] = roc_curve(Y_test, Y_pred_proba_test)

idx = np.min(np.where(tpr > 0.95)) # index of the first threshold for which the sensibility > 0.95
import matplotlib.pyplot as plt
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
X_val['Title'] = X_val['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
title_names = (X_val['Title'].value_counts() < 10)
X_val['Title'] = X_val['Title'].apply(lambda x: 'missing' if title_names.loc[x] == True else x)

X_val= apply_transformation(df=X_val, drop_list_woTarget=drop_list_woTarget)
#print(X_val.isna().sum())


# no_intersection(list(X_train.columns),list(X_val.columns))

df_val['Survived']= model.predict(X_val[var])

out=df_val[['PassengerId', 'Survived']]
out.to_csv(path_or_buf=full_path+"\\result.csv",index=False)