# -*- coding: utf-8 -*-

full_path="C:\\Users\\thoma\\Desktop\\VM share\\Python\\files\\Kaggle\\housing prices"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv(full_path+"\\train.csv", index_col='Id')

model_corr=df.corr()
high_corr=model_corr[abs(model_corr['SalePrice'])>0.1]
new_columns=high_corr.index
#independent=df[new_columns].drop(columns='SalePrice')

# select variables:    
all_columns=df.columns
independent=df[all_columns[:-1]]
dependent=df[all_columns[-1:]]

# check input data
detect_na=independent.isna().any()
issue=detect_na[detect_na == True].index
issue_cnt=independent.isnull().sum()
issue=issue_cnt[issue_cnt >= 0].index
no_issue=issue_cnt[issue_cnt == 0].index


# df.fillna({'weekday_checkins':0,
#            'weekend_checkins':0,
#            'average_tip_length':0,
#            'number_tips':0,
#            'average_caption_length':0,
#            'number_pics':0},
#           inplace=True)




for i in issue:
    print(df[i].value_counts())
    if df[i].dtypes != 'object':
        plt.hist(df[i])
        plt.title(i)
        plt.show()
        plt.clf()


print(df.info)
print(type(df))
print(df.dtypes)


#type dependent imputation
independent_input=independent[no_issue]
# independent_miss=independent.fillna('missing')
independent_dum=pd.get_dummies(independent_input)

independent_treated=independent_dum

# split test train data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(independent_treated,dependent,test_size=0.3, random_state=41, shuffle=True)

# modeling on train
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

model_param= list(zip(np.array(all_columns[:-1]),reg.coef_[0]))
independent_corr=independent.corr()

# assessing result
from sklearn.metrics import mean_squared_error, r2_score

#Train
print("R^2 Train score:")
print(reg.score(X_train, Y_train))
print('Mean squared error: %.2f'
      % mean_squared_error(Y_train, reg.predict(X_train)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(Y_train, reg.predict(X_train)))

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
