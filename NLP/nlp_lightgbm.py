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


from nltk.tokenize import word_tokenize
df['text_tokenized'] = df.apply(lambda row: word_tokenize(row['text']), axis=1)


### 4 pred data
# 4.1 
dependent=df['target']
independent=df.copy(deep=True)
independent.drop(columns='target',axis='columns', inplace=True)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(independent,dependent,test_size=0.3, random_state=41, shuffle=True)


# Bag of words
from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer()

# Define training_vectors:
training_vectors= bow_vectorizer.fit_transform(X_train.text).toarray()


import lightgbm as lgb

model = lgb.LGBMClassifier()

model.fit(training_vectors, Y_train)


testing_vectors= bow_vectorizer.transform(X_test.text).toarray()

Y_Pred=model.predict(testing_vectors)

confusion_matrix = pd.crosstab(Y_test, Y_Pred, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)




df_val=pd.read_csv(full_path+"\\test.csv", index_col='id')
val_vectors = bow_vectorizer.transform(df_val.text).toarray()
df_val['target']=model.predict(val_vectors)

#output
out=df_val['target']
out.to_csv(path_or_buf=full_path+"\\result.csv",index=True)

