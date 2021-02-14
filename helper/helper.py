# -*- coding: utf-8 -*-

# import os
# path="C:\\Users\\thoma\\Desktop\\VM share\\Python\\Kaggle\\helper"
# os.chdir(path)
# print(os.getcwd())
# %run helper.py

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


def treat_str_var(df,var):
    from sklearn.preprocessing import LabelEncoder # Converts cat data to numeric
    le=LabelEncoder()
    for i in var:
        df[i]=le.fit_transform(df[i])
