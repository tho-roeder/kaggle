# -*- coding: utf-8 -*-


def import_file(full_path):
    import pandas as pd
    df=pd.read_csv(full_path+"\\train.csv")
    return df

full_path="C:\\Users\\thoma\\Desktop\\VM share\\Python\\files\\Kaggle\\titanic"
df=import_file(full_path)

import os
path="C:\\Users\\thoma\\Desktop\\VM share\\Python\\Kaggle\\helper"
os.chdir(path)
print(os.chdir(path))
%run helper.py

#create new variables:
df['Cabin_new']=df['Cabin'].str[0:1]
plot_str_var(df,['Cabin_new'])

independent=df.drop(columns='Survived')
dependent=df['Survived']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(independent, dependent, train_size=0.3, random_state=41, shuffle=False)

X_train.drop(columns=['Cabin'], inplace=True)

all_var, num_var_nan, num_var_nonan, str_var_nan, str_var_nonan= pre_work(X_train)   

plot_num_var(X_train,num_var_nan)
plot_str_var(X_train,str_var_nan)

drop_list_num,impute_value_num=impute_var_v2(df=X_train,var=num_var_nan,perc_drop=0.25,style='median')
drop_list_str,impute_value_str=impute_var_v2(df=X_train,var=str_var_nan,perc_drop=1,style='nan')
drop_list_lowCor=merge_low_corr(df_ind=X_train, df_dep=Y_train, target='Survived', min_cor=0.05)
drop_list_max_perc_rep=same_value(df=X_train,var=all_var,max_perc_rep=0.95)


drop_list_woTarget=(drop_list_num+drop_list_str+drop_list_lowCor+drop_list_max_perc_rep+['Name','Ticket'])
# drop_list_woTarget.remove('SibSp')
drop_list=(drop_list_woTarget)


X_train.drop(columns=drop_list_woTarget, inplace=True)
import pandas as pd
X_train=pd.get_dummies(X_train)

# Standardize values
# SC=Standardize_values(X_train)
# X_train = SC.transform(X_train)


#Normalie values
# TF=Normalize_values(X_train)
# X_train = TF.transform(X_train)


def apply_transformation(df, drop_list_woTarget):
    df.drop(columns=['Cabin'], inplace=True)
    # impute_var(df=df,var=num_var_nan,perc_drop=1,style='median')
    # impute_var(df=df,var=num_var_nonan,perc_drop=1,style='median')
    # impute_var(df=df,var=str_var_nan,perc_drop=1,style='mode')
    df.drop(columns=drop_list_woTarget, inplace=True)
    df.dropna(axis=0, how='any', inplace=True)
    df=pd.get_dummies(df)
    df['Cabin_new_T']=0
    # df = SC.transform(df)
    # df=TF.transform(df)
    return df

X_test = apply_transformation(X_test, drop_list_woTarget)



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

   
# df_val=pd.read_csv(full_path+"\\test.csv")
# X_val=df_val.copy(deep=True)
# X_val['Cabin_new']=X_val['Cabin'].str[0:1]
# plot_str_var(X_val,['Cabin_new'])
# X_val= apply_transformation(df=X_val, drop_list_woTarget=drop_list_woTarget)

# df_val['Survived']= model.predict(X_val)

# out=df_val[['PassengerId', 'Survived']]
# out.to_csv(path_or_buf=full_path+"\\result.csv",index=False)


