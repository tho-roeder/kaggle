# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 19:12:57 2021

@author: thoma
"""

# for dataset in data_cleaner:
#     dataset['Family Size'] = dataset['SibSp'] + dataset['Parch'] + 1
#     dataset['IsAlone']= 1
#     dataset['IsAlone'].loc[dataset['Family Size'] > 1] = 0
    
#     dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
#     #Continuous variable bins - https://stackoverflow.com/questions/30211923/what-is-the-difference-between-pandas-qcut-and-pandas-cut
#     # using qcut  - https://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html
#     dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)
#     dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)
    
    
# stat_min = 10 
# title_names = (data1['Title'].value_counts() < stat_min)

   
# data1['Title'] = data1['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
# print(data1['Title'].value_counts())
# print("-"*10)


import pandas as pd


df['Fare_bin']=pd.qcut(df['Fare'], 10)

get_violinplot_for_target(df=df
               ,var=['Fare_bin'] 
               ,target='Survived')


def myfunc(n):
    if n <= 10.5:
        return 1
    elif n<=78:
        return 2
    else:
        return 3

df['Fare_bin2']=df['Fare'].apply(lambda x: myfunc(x))

get_violinplot_for_target(df=df
               ,var=['Fare_bin2'] 
               ,target='Survived')



from sklearn.tree import DecisionTreeClassifier

import numpy as np
#split=np.array(X_train['Fare']).reshape(-1, 1)

split=np.array(X_train['Age']).reshape(-1, 1)

classifier = DecisionTreeClassifier(random_state = 0,max_depth=2)
classifier.fit(split, Y_train)

from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(classifier
                ,out_file=dot_data
                ,filled=True
                ,rounded=True
                ,special_characters=True
                ,feature_names = ['Fare']
                ,proportion=True
                ,rotate=True
                ,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
# graph.write_png('tree.png')
Image(graph.create_png())

from sklearn.tree import _tree

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print ("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print ("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print ("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print ("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)
    
    
tree_to_code(tree=classifier, feature_names=["Age"])


def tree(Fare):
  if Fare <= 10.481249809265137:
    if Fare <= 7.133349895477295:
      return 1
    else:  # if Fare > 7.133349895477295
      return 2
  else:  # if Fare > 10.481249809265137
    if Fare <= 74.375:
      return 3
    else:  # if Fare > 74.375
      return 4
  
    
df['Fare_bin2']=df['Fare'].apply(lambda x: tree(x))

get_violinplot_for_target(df=df
               ,var=['Fare_bin2'] 
               ,target='Survived')


def tree(Age):
  if Age <= 40.25:
    if Age <= 17.5:
      return 1
    else:  # if Age > 17.5
      return 2
  else:  # if Age > 40.25
    if Age <= 58.5:
      return 3
    else:  # if Age > 58.5
      return 4

