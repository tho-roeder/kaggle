# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 13:45:23 2021

@author: thoma
"""

def import_file(full_path):
    import pandas as pd
    df=pd.read_csv(full_path+"\\train.csv", index_col='Id')
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


def treat_str_var(df,var):
    from sklearn.preprocessing import LabelEncoder # Converts cat data to numeric
    le=LabelEncoder()
    for i in var:
        df[i]=le.fit_transform(df[i])


def train_lightGBM(independent,dependent):
    import pandas as pd
    
    train_independent_dum=pd.get_dummies(independent)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(train_independent_dum, dependent, test_size = 0.3, random_state = 41)
    
    import lightgbm as lgb
    
    #Score 0.17203:
    d_train = lgb.Dataset(X_train, label=Y_train)
    params = {}
    params['learning_rate'] = 0.07
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'regression'
    params['metric'] = 'rmse'
    params['sub_feature'] = 0.5
    params['num_leaves'] = 10
    params['min_data'] = 50
    params['max_depth'] = 10
    
    #Score 0.18916:
    # params['learning_rate'] = 0.07
    # params['boosting_type'] = 'gbdt'
    # params['objective'] = 'regression'
    # params['metric'] = 'rmse'
    # params['sub_feature'] = 0.4
    # params['num_leaves'] = 10
    # params['min_data'] = 25
    # params['max_depth'] = 8
    
    model = lgb.train(params, d_train, 100)
    
    X_test['predict']=model.predict(X_test)
    final_df=X_test.merge(Y_test, on='Id')
    final_df['diff']=final_df['predict']-final_df['SalePrice']
    
    import matplotlib.pyplot as plt
    plt.scatter(final_df['predict'],final_df['SalePrice'])
    plt.show()
    
    from sklearn.metrics import mean_squared_error, r2_score
    
    #Train
    print('Mean squared error Train: %.2f'
          % mean_squared_error(Y_train, model.predict(X_train)))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination R2 Train: %.2f'
          % r2_score(Y_train, model.predict(X_train)))
    
    #Test
    print('Mean squared error Test: %.2f'
          % mean_squared_error(Y_test, model.predict(X_test, predict_disable_shape_check=True)))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination R2 Test: %.2f'
          % r2_score(Y_test, model.predict(X_test, predict_disable_shape_check=True)))
    return model


def create_output(model,drop_list,apply_trans):
    import pandas as pd
    #output
    test_df=pd.read_csv(full_path+"\\test.csv", index_col='Id')
    if apply_trans==True:
        # applying same transformation on test data results in less score
        # plot_str_var(test_df,str_var_nan)
        # plot_num_var(test_df,num_var_nan) 
        #test_drop_list_num=impute_var(df=test_df,var=num_var_nan,perc_drop=1,style='nan')
        impute_var(df=test_df,var=num_var_nan,perc_drop=1,style='nan')
        #test_drop_list_str=impute_var(df=test_df,var=str_var_nan,perc_drop=1,style='nan')
        impute_var(df=test_df,var=str_var_nan,perc_drop=1,style='nan')
        plot_str_var(test_df,str_var_nan)
        plot_num_var(test_df,num_var_nan) 
        test_df=test_df.drop(columns=drop_list)
    
    test_independent_dum=pd.get_dummies(test_df)
    test_df['SalePrice']= model.predict(test_independent_dum, predict_disable_shape_check=True)
    out=test_df['SalePrice']
    out.to_csv(path_or_buf=full_path+"\\result.csv",index=True)
    return test_df


full_path="C:\\Users\\thoma\\Desktop\\VM share\\Python\\files\\Kaggle\\housing prices"
df=import_file(full_path)
# print(len(df.columns))
# print(df.iloc[:,1])
# df=df[df['SalePrice']<=290000]

all_var, num_var_nan, num_var_nonan, str_var_nan, str_var_nonan= pre_work(df)   

# plot_num_var(df,num_var_nan)
# plot_str_var(df,str_var_nan)

drop_list_num=impute_var(df=df,var=num_var_nan,perc_drop=0.20,style='nan')
drop_list_str=impute_var(df=df,var=str_var_nan,perc_drop=1,style='nan')
#plot_num_var(df,num_var_nan)      
#plot_str_var(df,str_var_nan)

drop_list_lowCor=low_corr(df,target='SalePrice',min_cor=0.05)
drop_list_max_perc_rep=same_value(df,var=all_var,max_perc_rep=0.95)

drop_list=(drop_list_num+drop_list_str+drop_list_lowCor+drop_list_max_perc_rep)
#drop_list=drop_list.unique()
#str_list=str_var_nonan+str_var_nan
#treat_str_var(df,str_list)

all_columns=df.columns
independent=df[all_columns[:-1]].drop(columns=drop_list)
dependent=df[all_columns[-1:]]

model=train_lightGBM(independent,dependent)
test_df = create_output(model,drop_list,apply_trans=False)