# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 16:22:28 2021

@author: thoma
"""

### 1. import helper
import os
path="C:\\Users\\thoma\\Desktop\\VM share\\Python\\Kaggle\\helper"
os.chdir(path)
%run helper.py


### 2. source data
full_path="C:\\Users\\thoma\\Desktop\\VM share\\Python\\files\\Kaggle\\nlp-getting-started"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv(full_path+"\\train.csv", index_col='id')


### 3. analyse data
# 3.1 create overview of data set:
#getGeneralInformation(df)
all_var, num_var_nan, num_var_nonan, str_var_nan, str_var_nonan= pre_work(df)
value_counts=get_var_value_counts(df=df,var=str_var_nan)

### build model for location, regex etc.
#out=value_counts[1][2]

from nltk.tokenize import word_tokenize
df['text_tokenized'] = df.apply(lambda row: word_tokenize(row['text']), axis=1)

df['text_len'] =  df.apply(lambda row: len(row.text), axis=1)
df['text_cnt_elem'] =  df.apply(lambda row: len(row.text_tokenized), axis=1)



### 4 pred data
# 4.1 
dependent=df['target']
independent=df.copy(deep=True)
independent.drop(columns='target',axis='columns', inplace=True)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(independent,dependent,test_size=0.3, random_state=41, shuffle=True)


from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder


X_train.drop(columns='text_tokenized', inplace=True)
X_train.fillna('None', inplace=True)
column_trans = make_column_transformer(
    (CountVectorizer(), 'text'),
    (OneHotEncoder(handle_unknown='ignore'), str_var_nan),
    #(OrdinalEncoder(categories=category_array), categorial),
    remainder='passthrough')

X_train_trans=column_trans.fit_transform(X_train)
import lightgbm as lgb

model = lgb.LGBMClassifier()

model.fit(X_train_trans, Y_train)

X_test.drop(columns='text_tokenized', inplace=True)
X_test.fillna('None', inplace=True)
X_test_trans=column_trans.transform(X_test)

Y_Pred=model.predict(X_test_trans)

confusion_matrix = pd.crosstab(Y_test, Y_Pred, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)


df_val=pd.read_csv(full_path+"\\test.csv", index_col='id')
val_vectors = bow_vectorizer.transform(df_val.text).toarray()
df_val['target']=model.predict(val_vectors)

#output
out=df_val['target']
out.to_csv(path_or_buf=full_path+"\\result.csv",index=True)


# enter proximity
from country_list import countries_for_language
#Get country names in english and swedish
countries = dict(countries_for_language('en'))

us_states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", 
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]


