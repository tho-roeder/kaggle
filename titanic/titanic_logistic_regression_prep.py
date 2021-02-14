# -*- coding: utf-8 -*-

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


