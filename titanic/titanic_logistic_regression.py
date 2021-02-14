# -*- coding: utf-8 -*-

full_path="C:\\Users\\thoma\\Desktop\\VM share\\Python\\files\\Kaggle\\titanic"

import pandas as pd

df=pd.read_csv(full_path+"\\train.csv")

train_me=df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
lable_me=df['Survived']

train_me['Age']=train_me['Age'].fillna(train_me['Age'].mean())
train_me['Fare']=train_me['Fare'].fillna(train_me['Fare'].mean())
train_me=pd.get_dummies(train_me)


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(train_me, lable_me, train_size=0.3, random_state=41, shuffle=True)

from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
log.fit(X_train, Y_train)

from sklearn.metrics import classification_report
print(classification_report(Y_test, log.predict(X_test)))

from sklearn.metrics import mean_squared_error, r2_score
print('Mean squared error Train: %.2f'
      % mean_squared_error(Y_train, log.predict(X_train)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination R2 Train: %.2f'
      % r2_score(Y_train, log.predict(X_train)))

#Test
print('Mean squared error Test: %.2f'
      % mean_squared_error(Y_test, log.predict(X_test)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination R2 Test: %.2f'
      % r2_score(Y_test, log.predict(X_test)))

df=pd.read_csv(full_path+"\\test.csv")

train_me=df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

train_me['Age']=train_me['Age'].fillna(train_me['Age'].mean())
train_me['Fare']=train_me['Fare'].fillna(train_me['Fare'].mean())
train_me=pd.get_dummies(train_me)

df['Survived']= log.predict(train_me)

out=df[['PassengerId', 'Survived']]
out.to_csv(path_or_buf=full_path+"\\result.csv",index=False)


################################

df=pd.read_csv(full_path+"\\train.csv")

print(train_me.isna().any())

stats_me=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Ticket', 'Fare', 'Cabin', 'Embarked']
for i in stats_me:
    print(df[i].value_counts())

plot_me=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
import seaborn as sns
import matplotlib.pyplot as plt
for i in plot_me:
    sns.violinplot(x=df['Survived'], y=df[i])
    plt.show()
    plt.clf()

# save coeff
# calculated_coefficients = model.coef_ 
# intercept = model.intercept_

# print(calculated_coefficients )
# print(intercept )
#model_param= list(zip(np.array(all_columns[:-1]),reg.coef_[0]))

# Assign and update coefficients
# coefficients=model_2.coef_
# coefficients=coefficients.tolist()[0]



# import numpy as np
# from exam import hours_studied, calculated_coefficients, intercept

# def log_odds(features, coefficients,intercept):
#   return np.dot(features,coefficients) + intercept

# def sigmoid(z):
#     denominator = 1 + np.exp(-z)
#     return 1/denominator

# # Create predict_class() function here
# def predict_class(features, coefficients, intercept, threshold):
#   calculated_log_odds=log_odds(features, coefficients,intercept)
#   probabilities=sigmoid(calculated_log_odds)
#   return_value = np.where(probabilities >= threshold, 1, 0)
#   return return_value

# # Make final classifications on Codecademy University data here



# final_results=predict_class(hours_studied, calculated_coefficients, intercept, 0.5)
# print(final_results)