# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 21:01:29 2021

@author: thoma
"""

def import_file(full_path):
    import pandas as pd
    df=pd.read_csv(full_path+"\\train.csv")
    return df


def pre_work(df):
    num_var_nan=list()
    num_var_nonan=list()
    str_var_nan=list()
    str_var_nonan=list()
    for i in df.columns:
        if df[i].dtypes != 'object':
            if df[i].isna().sum()>0:
                num_var_nan.append(i)
            else:
                num_var_nonan.append(i)
        else:
            if df[i].isna().sum()>0:
                str_var_nan.append(i)
            else:
                str_var_nonan.append(i)
    return list(num_var_nan+ num_var_nonan+ str_var_nan+ str_var_nonan), num_var_nan, num_var_nonan, str_var_nan, str_var_nonan


def plot_num_var(df,var):
    import matplotlib.pyplot as plt
    for i in var:
        # print(df[i].value_counts())
        if df[i].dtypes != 'object':
            nbr_nan=df[i].isna().sum()
            perc_nan=nbr_nan/len(df[i])
            plt.hist(df[i])
            plt.title("Numeric var: {}, cnt: {}, perc: {}".format(i,nbr_nan,round(perc_nan,2)))
            plt.show()
            plt.clf()


def plot_str_var(df,var):
    import matplotlib.pyplot as plt
    import seaborn as sns
    for i in var:
        if df[i].dtypes == 'object':
            nbr_nan=df[i].isna().sum()
            perc_nan=nbr_nan/len(df[i])
            sns.countplot(x=df[i].fillna('Missing'),order = df[i].fillna('Missing').value_counts().index)
            plt.title("String var: {}, cnt: {}, perc: {}".format(i,nbr_nan,round(perc_nan,2)))
            plt.show()
            plt.clf()


def impute_var(df,var,perc_drop,style):
    import numpy as np
    var_drop=[]
    for i in var:
        if df[i].isna().sum()/len(df[i])>=perc_drop:
            var_drop.append(i)
        else:
            if df[i].dtypes != 'object':
                if style == 'mean':
                    df[i].fillna(value=df[i].mean(),inplace=True)
                if style == 'median':
                    df[i].fillna(value=df[i].median(),inplace=True)
                if style == 'nan':
                    df[i].fillna(value=np.nan,inplace=True)
            if df[i].dtypes == 'object':
                if style == 'nan':
                    df[i].fillna(value='missing',inplace=True)
            if style == 'mode':
                df[i].fillna(value=df[i].mode(dropna=True).values[0],inplace=True)
    return var_drop


def low_corr(df,target,min_cor):
    cor=df.corr()
    drop_list_lowCor=cor[abs(cor[target])<=min_cor]
    return list(drop_list_lowCor.index)


def same_value(df,var,max_perc_rep):
    drop_list_max_perc_rep=[]
    for i in var:
        if (df[i].value_counts().max()/len(df[i]))>=max_perc_rep:
            drop_list_max_perc_rep.append(i)
    return drop_list_max_perc_rep




full_path="C:\\Users\\thoma\\Desktop\\VM share\\Python\\files\\Kaggle\\titanic"
df=import_file(full_path)

df['Cabin_new']=df['Cabin'].str[0:1]
plot_str_var(df,['Cabin_new'])

all_var, num_var_nan, num_var_nonan, str_var_nan, str_var_nonan= pre_work(df)   

drop_list_num=impute_var(df=df,var=num_var_nan,perc_drop=0.20,style='median')
drop_list_str=impute_var(df=df,var=str_var_nan,perc_drop=1,style='mode')

drop_list_lowCor=low_corr(df,target='Survived',min_cor=0.05)
drop_list_max_perc_rep=same_value(df,var=all_var,max_perc_rep=0.95)
drop_list_woTarget=(drop_list_num+drop_list_str+drop_list_lowCor+drop_list_max_perc_rep+['Name','Ticket','Cabin'])
drop_list_woTarget.remove('SibSp')
drop_list=(drop_list_woTarget+['Survived'])

independent=df.drop(columns=drop_list)
dependent=df['Survived']

import pandas as pd
train_independent_dum=pd.get_dummies(independent)
   
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(train_independent_dum, dependent, train_size=0.3, random_state=41, shuffle=True)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=1000,solver='liblinear', random_state=41)
model.fit(X_train, Y_train)

from sklearn.metrics import mean_squared_error, r2_score
print('Mean squared error Train: %.2f'
      % mean_squared_error(Y_train, model.predict(X_train)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination R2 Train: %.2f'
      % r2_score(Y_train, model.predict(X_train)))

#Test
print('Mean squared error Test: %.2f'
      % mean_squared_error(Y_test, model.predict(X_test)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination R2 Test: %.2f'
      % r2_score(Y_test, model.predict(X_test)))

test_df=pd.read_csv(full_path+"\\test.csv")
test_df['Cabin_new']=test_df['Cabin'].str[0:1]
test_df['Cabin_new_T']=0
plot_str_var(test_df,['Cabin_new'])

test_df_dropped=test_df.drop(columns=drop_list_woTarget)
impute_var(df=test_df_dropped,var=num_var_nan,perc_drop=1,style='median')
num_var_nonan.remove('PassengerId')
num_var_nonan.remove('Survived')
num_var_nonan.remove('SibSp')

impute_var(df=test_df_dropped,var=num_var_nonan,perc_drop=1,style='median')


test_df_dropped=pd.get_dummies(test_df_dropped)

test_df['Survived']= model.predict(test_df_dropped)

out=test_df[['PassengerId', 'Survived']]
out.to_csv(path_or_buf=full_path+"\\result.csv",index=False)


