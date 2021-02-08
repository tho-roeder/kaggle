# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 15:10:04 2021

@author: thoma
"""
# try normalization and standardization

import pandas as pd
import os

df=pd.read_csv(os.getcwd()+"\\Desktop\\VM share\\Python\\Kaggle\\titanic - logistic regression"+"\\train.csv")

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

df=pd.read_csv(os.getcwd()+"\\Desktop\\VM share\\Python\\Kaggle\\titanic"+"\\test.csv")

train_me=df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

train_me['Age']=train_me['Age'].fillna(train_me['Age'].mean())
train_me['Fare']=train_me['Fare'].fillna(train_me['Fare'].mean())
train_me=pd.get_dummies(train_me)

df['Survived']= log.predict(train_me)

out=df[['PassengerId', 'Survived']]
out.to_csv(path_or_buf=os.getcwd()+"\\Desktop\\VM share\\Python\\Kaggle\\titanic"+"\\result.csv",index=False)


################################

df=pd.read_csv(os.getcwd()+"\\Desktop\\VM share\\Python\\Kaggle\\titanic"+"\\train.csv")

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

#convert: 'Cabin'
# print(df['Cabin'])
# print(df['Cabin'].value_counts())
# check=df['Cabin'].dropna()
# check[:1]
#check=df['Cabin'].fillna('Missing')